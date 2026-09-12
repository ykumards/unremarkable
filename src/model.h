#ifndef UNREMARKABLE_SRC_MODEL_H_
#define UNREMARKABLE_SRC_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace unremarkable {

struct Config {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  float rope_theta = 10000.0f;

  int head_size() const { return dim / n_heads; }
  int kv_dim() const { return head_size() * n_kv_heads; }
  size_t cache_elements(int context) const;
};

// A read-only view, not another allocation. Row r starts at data + r * columns.
struct Matrix {
  const float* data = nullptr;
  int rows = 0;
  int columns = 0;
};

struct LayerWeights {
  std::span<const float> attention_norm;     // [dim]
  std::span<const float> feed_forward_norm;  // [dim]
  Matrix query;                              // [dim, dim]
  Matrix key;                                // [kv_dim, dim]
  Matrix value;                              // [kv_dim, dim]
  Matrix attention_output;                   // [dim, dim]
  Matrix gate;                               // [hidden_dim, dim]
  Matrix up;                                 // [hidden_dim, dim]
  Matrix down;                               // [dim, hidden_dim]
};

// Owns one resident FP32 checkpoint. All views borrow from its data_ buffer.
// Config and views are initialized at load time and treated as immutable.
class Model {
 public:
  explicit Model(const std::string& checkpoint);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Config config{};
  Matrix embedding;
  std::vector<LayerWeights> layers;
  std::span<const float> final_norm;
  Matrix classifier;
  size_t weight_bytes() const { return data_.size() * sizeof(float); }

 private:
  // Validate total size before allocating, then bind the views after reading.
  size_t weight_layout(bool shared, bool legacy);
  std::vector<float> data_;
};

}  // namespace unremarkable
#endif
