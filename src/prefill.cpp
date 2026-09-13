#include <algorithm>
#include <stdexcept>

#include "engine.h"
#include "kernels.h"

namespace unremarkable {

// X[count, columns] * W[rows, columns]^T -> output[count, rows].
void Engine::project_batch(const float* input, const Matrix& weight, int count, float* output) {
  project_group(input, count, {{weight, output}});
}

void Engine::prefill_chunk(std::span<const int> tokens) {
  const int count = static_cast<int>(tokens.size());
  const int dim = config_.dim;
  const int hidden = config_.hidden_dim;
  const int kv_dim = config_.kv_dim();
  const int head_size = config_.head_size();
  const int start = next_position_;

  float* residual = scratch_.residual.data();
  float* normalized = scratch_.normalized.data();
  float* query = scratch_.query.data();
  float* attention_output = scratch_.attention_output.data();
  float* projected = scratch_.projected.data();
  float* gate = scratch_.gate.data();
  float* up = scratch_.up.data();

  for (int token = 0; token < count; ++token) {
    embedding_lookup(model_.embedding, tokens[token], residual + static_cast<size_t>(token) * dim);
  }
  profiler_.lap(Stage::kEmbedding);
  for (int token = 0; token < count; ++token) {
    float* cosines = scratch_.rope.data() + static_cast<size_t>(token) * head_size;
    rope_angles(start + token, head_size, config_.rope_theta, cosines, cosines + head_size / 2);
  }
  profiler_.lap(Stage::kRope);
  for (int layer = 0; layer < config_.n_layers; ++layer) {
    const LayerWeights& weights = model_.layers[layer];
    float* key_history = cache_.keys_for_layer(layer);
    float* value_history = cache_.values_for_layer(layer);
    float* keys = key_history + static_cast<size_t>(start) * kv_dim;
    float* values = value_history + static_cast<size_t>(start) * kv_dim;

    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      rmsnorm(residual + offset, weights.attention_norm.data(), dim, normalized + offset);
    }
    profiler_.lap(Stage::kNorm);
    project_group(normalized, count,
                  {{weights.query, query}, {weights.key, keys}, {weights.value, values}});
    profiler_.lap(Stage::kQkv);
    for (int token = 0; token < count; ++token) {
      const float* cosines = scratch_.rope.data() + static_cast<size_t>(token) * head_size;
      rope_inplace(cosines, cosines + head_size / 2, head_size, dim, kv_dim,
                   query + static_cast<size_t>(token) * dim,
                   keys + static_cast<size_t>(token) * kv_dim);
    }
    profiler_.lap(Stage::kRope);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      // All K/V slots exist, but each query only reads through its own position.
      attend(query + offset, key_history, value_history, start + token, attention_output + offset);
    }
    profiler_.lap(Stage::kAttention);
    project_batch(attention_output, weights.attention_output, count, projected);
    profiler_.lap(Stage::kOutput);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      add_inplace(projected + offset, dim, residual + offset);
      rmsnorm(residual + offset, weights.feed_forward_norm.data(), dim, normalized + offset);
    }
    profiler_.lap(Stage::kNorm);
    project_group(normalized, count, {{weights.gate, gate}, {weights.up, up}});
    profiler_.lap(Stage::kGateUp);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * hidden;
      swiglu_inplace(up + offset, hidden, gate + offset);
    }
    profiler_.lap(Stage::kSwiGLU);
    project_batch(gate, weights.down, count, projected);
    profiler_.lap(Stage::kDown);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      add_inplace(projected + offset, dim, residual + offset);
    }
    profiler_.lap(Stage::kResidual);
  }
  next_position_ += count;
}

std::span<float> Engine::prefill(std::span<const int> tokens, int batch_size) {
  if (batch_size < 1 || batch_size > kMaxPrefillBatch) {
    throw std::runtime_error("prefill batch must be between 1 and 8");
  }
  if (tokens.empty() || tokens.size() > static_cast<size_t>(config_.seq_len - next_position_)) {
    throw std::runtime_error("prefill must contain tokens and fit the remaining context");
  }
  for (int token : tokens) {
    if (token < 0 || token >= config_.vocab_size) {
      throw std::runtime_error("token ID out of range");
    }
  }
  profiler_.begin();
  int last_count = 0;
  for (size_t offset = 0; offset < tokens.size(); offset += last_count) {
    last_count =
        static_cast<int>(std::min(tokens.size() - offset, static_cast<size_t>(batch_size)));
    prefill_chunk(tokens.subspan(offset, last_count));
  }
  // Only the final prompt token needs vocabulary scores. Earlier logits are unused.
  float* last = scratch_.residual.data() + static_cast<size_t>(last_count - 1) * config_.dim;
  rmsnorm(last, model_.final_norm.data(), config_.dim, last);
  profiler_.lap(Stage::kNorm);
  project(last, model_.classifier, scratch_.logits.data());
  profiler_.lap(Stage::kClassifier);
  profiler_.end(static_cast<int>(tokens.size()));
  return scratch_.logits;
}

}  // namespace unremarkable
