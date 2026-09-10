// Dense FP32 model, inference buffers, and sampling.
#ifndef UNREMARKABLE_SRC_ENGINE_H_
#define UNREMARKABLE_SRC_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kernels.h"
#include "pool.h"

namespace unremarkable {

enum class Quantization : int32_t { kFloat32 = 0, kQ8_0 = 1 };

struct Config {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  float rope_theta = 10000.0f;                         // Legacy checkpoints predate the field.
  Quantization quantization = Quantization::kFloat32;  // Version 2 onwards.
};

// Loads an FP32 checkpoint, either legacy llama2.c or the tagged format written
// by tools/export_hf.py, and owns resident weights and sequence state.
class Engine {
 public:
  Engine(const std::string& checkpoint, int context = 0, int threads = 1);
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const Config& config() const { return config_; }
  size_t weight_bytes() const { return data_.size(); }
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

  // Bytes one row of `columns` values occupies in this checkpoint's format.
  size_t row_bytes(int columns) const;
  // Quantizes `input` if required, then projects `rows` rows from `base`.
  void project(const float* input, const std::byte* base, int columns, int rows, float* output);

  // Matrices follow the checkpoint's weight format and are addressed in bytes;
  // norm vectors are always FP32.
  struct Weights {
    const std::byte* token_embedding_table = nullptr;  // [vocabulary][dim]
    const float* rms_att_weight = nullptr;             // [layer][dim]
    const float* rms_ffn_weight = nullptr;             // [layer][dim]
    const std::byte* wq = nullptr;                     // [layer][dim][dim]
    const std::byte* wk = nullptr;                     // [layer][kv_dim][dim]
    const std::byte* wv = nullptr;                     // [layer][kv_dim][dim]
    const std::byte* wo = nullptr;                     // [layer][dim][dim]
    const std::byte* w1 = nullptr;                     // [layer][hidden_dim][dim]
    const std::byte* w2 = nullptr;                     // [layer][dim][hidden_dim]
    const std::byte* w3 = nullptr;                     // [layer][hidden_dim][dim]
    const float* rms_final_weight = nullptr;           // [dim]
    const std::byte* wcls = nullptr;                   // [vocabulary][dim]
  } weights_;

  struct State {
    std::vector<float> x;            // residual stream [dim]
    std::vector<float> xb;           // normalized input / attention output [dim]
    std::vector<float> xb2;          // attention projection [dim]
    std::vector<float> hb;           // feed-forward gate [hidden_dim]
    std::vector<float> hb2;          // feed-forward up projection [hidden_dim]
    std::vector<float> q;            // query [dim]
    std::vector<Q8Block> xq;         // quantized projection input, when quantized
    std::vector<float> att;          // attention scores [head][context]
    std::vector<float> logits;       // [vocabulary]
    std::vector<float> key_cache;    // [layer][context][kv_dim]
    std::vector<float> value_cache;  // [layer][context][kv_dim]
  } state_;

  Config config_{};
  Pool pool_;
  std::vector<std::byte> data_;
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
