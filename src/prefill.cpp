#include <algorithm>
#include <stdexcept>

#include "engine.h"
#include "kernels.h"

namespace unremarkable {

// X[count, columns] * W[rows, columns]^T -> output[count, rows].
void Engine::project_batch(const float* input, const Matrix& weight, int count, float* output) {
  if (count == 1) {
    project(input, weight, output);
    return;
  }
  const bool quantized = weight.format == Format::kQ8_0;
  if (quantized) {
    const int blocks = q8_blocks(weight.columns);
    for (int token = 0; token < count; ++token) {
      quantize_q8(input + static_cast<size_t>(token) * weight.columns, weight.columns,
                  scratch_.quantized.data() + static_cast<size_t>(token) * blocks);
    }
  }
  const size_t stride = row_bytes(weight.format, weight.columns);
  worker_.run(weight.rows, [&](int begin, int end) {
    Matrix rows{weight.data + static_cast<size_t>(begin) * stride, end - begin, weight.columns,
                weight.format};
    if (quantized) {
      matmul_q8(scratch_.quantized.data(), rows, count, weight.rows, output + begin);
    } else {
      matmul(input, rows, count, weight.rows, output + begin);
    }
  });
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
    project_batch(normalized, weights.query, count, query);
    project_batch(normalized, weights.key, count, keys);
    project_batch(normalized, weights.value, count, values);
    for (int token = 0; token < count; ++token) {
      rope_inplace(start + token, head_size, dim, kv_dim, config_.rope_theta,
                   query + static_cast<size_t>(token) * dim,
                   keys + static_cast<size_t>(token) * kv_dim);
    }
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      // All K/V slots exist, but each query only reads through its own position.
      causal_attention(query + offset, key_history, value_history, config_.n_heads,
                       config_.n_kv_heads, head_size, config_.seq_len, start + token,
                       scratch_.attention_scores.data(), attention_output + offset);
    }
    project_batch(attention_output, weights.attention_output, count, projected);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      add_inplace(projected + offset, dim, residual + offset);
      rmsnorm(residual + offset, weights.feed_forward_norm.data(), dim, normalized + offset);
    }
    project_batch(normalized, weights.gate, count, gate);
    project_batch(normalized, weights.up, count, up);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * hidden;
      swiglu_inplace(up + offset, hidden, gate + offset);
    }
    project_batch(gate, weights.down, count, projected);
    for (int token = 0; token < count; ++token) {
      const size_t offset = static_cast<size_t>(token) * dim;
      add_inplace(projected + offset, dim, residual + offset);
    }
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
  int last_count = 0;
  for (size_t offset = 0; offset < tokens.size(); offset += last_count) {
    last_count =
        static_cast<int>(std::min(tokens.size() - offset, static_cast<size_t>(batch_size)));
    prefill_chunk(tokens.subspan(offset, last_count));
  }
  // Only the final prompt token needs vocabulary scores. Earlier logits are unused.
  float* last = scratch_.residual.data() + static_cast<size_t>(last_count - 1) * config_.dim;
  rmsnorm(last, model_.final_norm.data(), config_.dim, last);
  project(last, model_.classifier, scratch_.logits.data());
  return scratch_.logits;
}

}  // namespace unremarkable
