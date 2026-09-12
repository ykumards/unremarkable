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
    : residual(config.dim),
      normalized(config.dim),
      query(config.dim),
      attention_output(config.dim),
      projected(config.dim),
      gate(config.hidden_dim),
      up(config.hidden_dim),
      attention_scores(static_cast<size_t>(config.n_heads) * config.seq_len),
      logits(config.vocab_size),
      quantized(config.format == Format::kQ8_0 ? q8_blocks(std::max(config.dim, config.hidden_dim))
                                               : 0) {}

Engine::Engine(const std::string& checkpoint, int context)
    : model_(checkpoint),
      config_(with_context(model_.config, context)),
      cache_(config_, config_.seq_len),
      scratch_(config_) {}

void Engine::reset() {
  // Old cache entries need no clearing: forward overwrites each position before
  // attention can read it. Weights and all working allocations remain alive.
  next_position_ = 0;
}

void Engine::project(const float* input, const Matrix& weight, float* output) {
  if (weight.format == Format::kQ8_0) {
    quantize_q8(input, weight.columns, scratch_.quantized.data());
    matvec_q8(scratch_.quantized.data(), weight, output);
  } else {
    matvec(input, weight, output);
  }
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
  embedding_lookup(model_.embedding, token, residual);

  for (int layer = 0; layer < config_.n_layers; ++layer) {
    const LayerWeights& weights = model_.layers[layer];
    float* key_history = cache_.keys_for_layer(layer);
    float* value_history = cache_.values_for_layer(layer);
    float* key = key_history + static_cast<size_t>(position) * kv_dim;
    float* value = value_history + static_cast<size_t>(position) * kv_dim;

    // 2. Attention. K/V write directly into this token's persistent cache slots.
    rmsnorm(residual, weights.attention_norm.data(), dim, normalized);
    project(normalized, weights.query, query);
    project(normalized, weights.key, key);
    project(normalized, weights.value, value);
    rope_inplace(position, head_size, dim, kv_dim, config_.rope_theta, query, key);
    causal_attention(query, key_history, value_history, config_.n_heads, config_.n_kv_heads,
                     head_size, config_.seq_len, position, scratch_.attention_scores.data(),
                     attention_output);
    project(attention_output, weights.attention_output, projected);
    add_inplace(projected, dim, residual);

    // 3. Feed-forward. Expand to [hidden_dim], gate, project back to [dim].
    rmsnorm(residual, weights.feed_forward_norm.data(), dim, normalized);
    project(normalized, weights.gate, gate);
    project(normalized, weights.up, up);
    swiglu_inplace(up, config_.hidden_dim, gate);
    project(gate, weights.down, projected);
    add_inplace(projected, dim, residual);
  }

  // 4. Final activation -> one score for every possible next token.
  rmsnorm(residual, model_.final_norm.data(), dim, residual);
  project(residual, model_.classifier, scratch_.logits.data());
  ++next_position_;
  return scratch_.logits;
}

}  // namespace unremarkable
