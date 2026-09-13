#ifndef UNREMARKABLE_SRC_ENGINE_H_
#define UNREMARKABLE_SRC_ENGINE_H_

#include <span>
#include <string>
#include <vector>

#include "model.h"
#include "profile.h"
#include "worker.h"

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

inline constexpr int kMaxPrefillBatch = 8;

// Working memory for up to eight tokens. Decode reuses the first token slot.
struct Scratch {
  explicit Scratch(const Config& config);
  std::vector<float> residual;          // [batch, dim], carried through all layers
  std::vector<float> normalized;        // [batch, dim]
  std::vector<float> query;             // [batch, dim]
  std::vector<float> attention_output;  // [batch, dim]
  std::vector<float> projected;         // [batch, dim], reused for both residual additions
  std::vector<float> gate;              // [batch, hidden_dim], overwritten by SwiGLU
  std::vector<float> up;                // [batch, hidden_dim]
  std::vector<float> attention_scores;  // [n_heads, context]
  std::vector<float> logits;            // [vocab_size]
  std::vector<Q8Block> quantized;       // projection input, for Q8_0 weights only
};

class Engine {
 public:
  explicit Engine(const std::string& checkpoint, int context = 0, int threads = 1);
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const Config& config() const { return config_; }
  size_t weight_bytes() const { return model_.weight_bytes(); }
  int threads() const { return worker_.threads(); }
  size_t kv_bytes() const { return cache_.bytes(); }

  // One token in, next-token scores out. Positions must be consecutive from 0.
  // The returned span borrows scratch_.logits; forward or prefill overwrites it.
  std::span<float> forward(int token, int position);

  // Append prompt tokens to the cached prefix. Returns only the final logits.
  // The returned span borrows the same buffer as forward().
  // Rejects empty inputs, invalid tokens, and context overflow before changing state.
  std::span<float> prefill(std::span<const int> tokens, int batch_size = kMaxPrefillBatch);
  void reset();

#ifdef UNREMARKABLE_PROFILE
  // Totals since construction; subtract two snapshots for one phase.
  Profile profile() const;
#endif

 private:
  // output = weight * input, quantizing the input first for Q8_0 weights.
  void project(const float* input, const Matrix& weight, float* output);

  void project_batch(const float* input, const Matrix& weight, int count, float* output);
  void prefill_chunk(std::span<const int> tokens);

  Model model_;       // Permanent learned weights.
  Config config_;     // Model dimensions, with the selected context capacity.
  KVCache cache_;     // History of the current sequence.
  Scratch scratch_;   // Temporary calculations for a token or prompt chunk.
  RowWorker worker_;  // Joined before scratch and model storage are destroyed.
  int next_position_ = 0;
  [[no_unique_address]] Profiler profiler_;  // Empty unless -DUNREMARKABLE_PROFILE.
};

}  // namespace unremarkable
#endif
