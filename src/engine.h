#ifndef UNREMARKABLE_SRC_ENGINE_H_
#define UNREMARKABLE_SRC_ENGINE_H_

#include <span>
#include <string>
#include <vector>

#include "model.h"

namespace unremarkable {

// Sequence history. Each layer retains one key and value per processed token.
struct KVCache {
  KVCache(const Config& config, int context);
  float* keys_for_layer(int layer);
  float* values_for_layer(int layer);
  size_t bytes() const { return (keys.size() + values.size()) * sizeof(float); }

  int context;
  int kv_dim;
  std::vector<float> keys;    // [layer, context, kv_dim]
  std::vector<float> values;  // [layer, context, kv_dim]
};

// One token's working memory. Allocated once and overwritten in forward().
struct Scratch {
  explicit Scratch(const Config& config);
  std::vector<float> residual;          // [dim], carried through all layers
  std::vector<float> normalized;        // [dim]
  std::vector<float> query;             // [dim]
  std::vector<float> attention_output;  // [dim]
  std::vector<float> projected;         // [dim], reused for both residual additions
  std::vector<float> gate;              // [hidden_dim], overwritten by SwiGLU
  std::vector<float> up;                // [hidden_dim]
  std::vector<float> attention_scores;  // [n_heads, context]
  std::vector<float> logits;            // [vocab_size]
  std::vector<Q8Block> quantized;       // projection input, for Q8_0 weights only
};

class Engine {
 public:
  explicit Engine(const std::string& checkpoint, int context = 0);
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const Config& config() const { return config_; }
  size_t weight_bytes() const { return model_.weight_bytes(); }
  size_t kv_bytes() const { return cache_.bytes(); }

  // One token in, next-token scores out. Positions must be consecutive from 0.
  // The returned span borrows scratch_.logits; the next forward overwrites it.
  std::span<float> forward(int token, int position);
  void reset();

 private:
  // output = weight * input, quantizing the input first for Q8_0 weights.
  void project(const float* input, const Matrix& weight, float* output);

  Model model_;      // Permanent learned weights.
  Config config_;    // Model dimensions, with the selected context capacity.
  KVCache cache_;    // History of the current sequence.
  Scratch scratch_;  // Temporary calculations for the current token.
  int next_position_ = 0;
};

}  // namespace unremarkable
#endif
