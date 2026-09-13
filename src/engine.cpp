// Forward pass adapted from karpathy/llama2.c. See THIRD_PARTY.md.
#include "engine.h"

#include <algorithm>
#include <stdexcept>

#include "kernels.h"

namespace unremarkable {
namespace {

Config with_context(Config config, int context) {
  if (context < 0 || context > config.seq_len) {
    throw std::runtime_error("requested context exceeds model context");
  }
  if (context) {
    config.seq_len = context;
  }
  return config;
}

}  // namespace

KVCache::KVCache(const Config& config, int capacity)
    : context(capacity),
      kv_dim(config.kv_dim()),
      keys(config.cache_elements(capacity)),
      values(config.cache_elements(capacity)) {}

float* KVCache::keys_for_layer(int layer) {
  return keys.data() + static_cast<size_t>(layer) * context * kv_dim;
}

float* KVCache::values_for_layer(int layer) {
  return values.data() + static_cast<size_t>(layer) * context * kv_dim;
}

Scratch::Scratch(const Config& config)
    : residual(static_cast<size_t>(kMaxPrefillBatch) * config.dim),
      normalized(static_cast<size_t>(kMaxPrefillBatch) * config.dim),
      query(static_cast<size_t>(kMaxPrefillBatch) * config.dim),
      attention_output(static_cast<size_t>(kMaxPrefillBatch) * config.dim),
      projected(static_cast<size_t>(kMaxPrefillBatch) * config.dim),
      gate(static_cast<size_t>(kMaxPrefillBatch) * config.hidden_dim),
      up(static_cast<size_t>(kMaxPrefillBatch) * config.hidden_dim),
      attention_scores(static_cast<size_t>(config.n_heads) * config.seq_len),
      logits(config.vocab_size),
      quantized(config.format == Format::kQ8_0
                    ? static_cast<size_t>(kMaxPrefillBatch) *
                          q8_blocks(std::max(config.dim, config.hidden_dim))
                    : 0),
      rope(static_cast<size_t>(kMaxPrefillBatch) * config.head_size()) {}

Engine::Engine(const std::string& checkpoint, int context, int threads)
    : model_(checkpoint),
      config_(with_context(model_.config, context)),
      cache_(config_, config_.seq_len),
      scratch_(config_),
      worker_(threads) {}

void Engine::reset() {
  // Old cache entries need no clearing: forward overwrites each position before
  // attention can read it. Weights and all working allocations remain alive.
  next_position_ = 0;
}

#ifdef UNREMARKABLE_PROFILE
Profile Engine::profile() const {
  Profile totals = profiler_.totals();
  totals.waiting = worker_.waiting_seconds();
  totals.worker = worker_.busy_seconds();
  return totals;
}
#endif

void Engine::project(const float* input, const Matrix& weight, float* output) {
  project_group(input, 1, {{weight, output}});
}

void Engine::project_group(const float* input, int count, std::initializer_list<Projection> group) {
  const Matrix& first = group.begin()->weight;
  const bool quantized = first.format == Format::kQ8_0;
  if (quantized) {
    const double started = profiler_.clock();
    const int blocks = q8_blocks(first.columns);
    for (int token = 0; token < count; ++token) {
      quantize_q8(input + static_cast<size_t>(token) * first.columns, first.columns,
                  scratch_.quantized.data() + static_cast<size_t>(token) * blocks);
    }
    profiler_.quantized(started);
  }
  int total_rows = 0;
  for (const auto& projection : group) {
    total_rows += projection.weight.rows;
  }
  const size_t stride = row_bytes(first.format, first.columns);
  worker_.run(total_rows, [&](int begin, int end) {
    int offset = 0;
    for (const auto& [weight, output] : group) {
      const int row_begin = std::max(0, begin - offset);
      const int row_end = std::min(weight.rows, end - offset);
      if (row_begin < row_end) {
        Matrix rows{weight.data + static_cast<size_t>(row_begin) * stride, row_end - row_begin,
                    weight.columns, weight.format};
        if (count == 1) {
          if (quantized) {
            matvec_q8(scratch_.quantized.data(), rows, output + row_begin);
          } else {
            matvec(input, rows, output + row_begin);
          }
        } else if (quantized) {
          matmul_q8(scratch_.quantized.data(), rows, count, weight.rows, output + row_begin);
        } else {
          matmul(input, rows, count, weight.rows, output + row_begin);
        }
      }
      offset += weight.rows;
    }
  });
}

void Engine::attend(const float* query, const float* keys, const float* values, int position,
                    float* output) {
  // Split whole heads; their score and output slices do not overlap.
  worker_.run(
      config_.n_heads,
      [&](int begin, int end) {
        causal_attention(query, keys, values, config_.n_heads, config_.n_kv_heads,
                         config_.head_size(), config_.seq_len, position, begin, end,
                         scratch_.attention_scores.data(), output);
      },
      2);
}

std::span<float> Engine::forward(int token, int position) {
  if (token < 0 || token >= config_.vocab_size) {
    throw std::runtime_error("token ID out of range");
  }
  if (position != next_position_ || position < 0 || position >= config_.seq_len) {
    throw std::runtime_error("forward position must follow the cached prefix and fit context");
  }
  const int dim = config_.dim;
  const int kv_dim = config_.kv_dim();
  const int head_size = config_.head_size();

  // Reuse names from the scratch buffer
  float* residual = scratch_.residual.data();
  float* normalized = scratch_.normalized.data();
  float* query = scratch_.query.data();
  float* attention_output = scratch_.attention_output.data();
  float* projected = scratch_.projected.data();
  float* gate = scratch_.gate.data();
  float* up = scratch_.up.data();

  // 1. Token ID -> one embedding row -> the running FP32 activation [dim].
  profiler_.begin();
  embedding_lookup(model_.embedding, token, residual);
  profiler_.lap(Stage::kEmbedding);
  const float* cosines = scratch_.rope.data();
  const float* sines = cosines + head_size / 2;
  rope_angles(position, head_size, config_.rope_theta, scratch_.rope.data(),
              scratch_.rope.data() + head_size / 2);
  profiler_.lap(Stage::kRope);

  for (int layer = 0; layer < config_.n_layers; ++layer) {
    const LayerWeights& weights = model_.layers[layer];
    float* key_history = cache_.keys_for_layer(layer);
    float* value_history = cache_.values_for_layer(layer);
    float* key = key_history + static_cast<size_t>(position) * kv_dim;
    float* value = value_history + static_cast<size_t>(position) * kv_dim;

    // 2. Attention. K/V write directly into this token's persistent cache slots.
    rmsnorm(residual, weights.attention_norm.data(), dim, normalized);
    profiler_.lap(Stage::kNorm);
    project_group(normalized, 1,
                  {{weights.query, query}, {weights.key, key}, {weights.value, value}});
    profiler_.lap(Stage::kQkv);
    rope_inplace(cosines, sines, head_size, dim, kv_dim, query, key);
    profiler_.lap(Stage::kRope);
    attend(query, key_history, value_history, position, attention_output);
    profiler_.lap(Stage::kAttention);
    project(attention_output, weights.attention_output, projected);
    profiler_.lap(Stage::kOutput);
    add_inplace(projected, dim, residual);
    profiler_.lap(Stage::kResidual);

    // 3. Feed-forward. Expand to [hidden_dim], gate, project back to [dim].
    rmsnorm(residual, weights.feed_forward_norm.data(), dim, normalized);
    profiler_.lap(Stage::kNorm);
    project_group(normalized, 1, {{weights.gate, gate}, {weights.up, up}});
    profiler_.lap(Stage::kGateUp);
    swiglu_inplace(up, config_.hidden_dim, gate);
    profiler_.lap(Stage::kSwiGLU);
    project(gate, weights.down, projected);
    profiler_.lap(Stage::kDown);
    add_inplace(projected, dim, residual);
    profiler_.lap(Stage::kResidual);
  }

  // 4. Final activation -> one score for every possible next token.
  rmsnorm(residual, model_.final_norm.data(), dim, residual);
  profiler_.lap(Stage::kNorm);
  project(residual, model_.classifier, scratch_.logits.data());
  profiler_.lap(Stage::kClassifier);
  ++next_position_;
  profiler_.end();
  return scratch_.logits;
}

}  // namespace unremarkable
