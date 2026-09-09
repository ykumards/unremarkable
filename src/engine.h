// Dense FP32 model, inference buffers, and sampling.
#ifndef UNREMARKABLE_SRC_ENGINE_H_
#define UNREMARKABLE_SRC_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace unremarkable {

struct Config {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  float rope_theta = 10000.0f;  // Legacy checkpoints predate the field.
};

// Loads an FP32 checkpoint, either legacy llama2.c or the tagged format written
// by tools/export_hf.py, and owns resident weights and sequence state.
class Engine {
 public:
  explicit Engine(const std::string& checkpoint, int context = 0);
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const Config& config() const { return config_; }
  size_t weight_bytes() const { return data_.size() * sizeof(float); }
  size_t kv_bytes() const;

  // Positions must be consecutive, starting at zero. The returned logits borrow
  // engine storage, remain valid until destruction, and change on each forward.
  std::span<float> forward(int token, int position);
  void reset();

 private:
  // Walks the fixed tensor order. With data_ empty it validates shapes and returns
  // the expected payload size; once data_ is sized it binds the weight views. Legacy
  // checkpoints carry two unused RoPE tables that the tagged format omits.
  size_t weight_layout(bool shared, bool legacy);

  struct Weights {
    const float* token_embedding_table = nullptr;  // [vocabulary][dim]
    const float* rms_att_weight = nullptr;         // [layer][dim]
    const float* rms_ffn_weight = nullptr;         // [layer][dim]
    const float* wq = nullptr;                     // [layer][dim][dim]
    const float* wk = nullptr;                     // [layer][kv_dim][dim]
    const float* wv = nullptr;                     // [layer][kv_dim][dim]
    const float* wo = nullptr;                     // [layer][dim][dim]
    const float* w1 = nullptr;                     // [layer][hidden_dim][dim]
    const float* w2 = nullptr;                     // [layer][dim][hidden_dim]
    const float* w3 = nullptr;                     // [layer][hidden_dim][dim]
    const float* rms_final_weight = nullptr;       // [dim]
    const float* wcls = nullptr;                   // [vocabulary][dim]
  } weights_;

  struct State {
    std::vector<float> x;            // residual stream [dim]
    std::vector<float> xb;           // normalized input / attention output [dim]
    std::vector<float> xb2;          // attention projection [dim]
    std::vector<float> hb;           // feed-forward gate [hidden_dim]
    std::vector<float> hb2;          // feed-forward up projection [hidden_dim]
    std::vector<float> q;            // query [dim]
    std::vector<float> att;          // attention scores [head][context]
    std::vector<float> logits;       // [vocabulary]
    std::vector<float> key_cache;    // [layer][context][kv_dim]
    std::vector<float> value_cache;  // [layer][context][kv_dim]
  } state_;

  Config config_{};
  std::vector<float> data_;
  int next_position_ = 0;
};

class Sampler {
 public:
  Sampler(int vocabulary, float temperature = 0, float top_p = 0.9f, uint64_t seed = 1);
  // Temperature sampling replaces logits with probabilities in place.
  int sample(std::span<float> logits);

 private:
  struct Candidate {
    float probability;
    int token;
  };
  float random_float();
  std::vector<Candidate> candidates_;
  float temperature_;
  float top_p_;
  uint64_t rng_state_;
};

}  // namespace unremarkable
#endif
