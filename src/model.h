#ifndef UNREMARKABLE_SRC_MODEL_H_
#define UNREMARKABLE_SRC_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace unremarkable {

// How a checkpoint stores its matrices. Norm vectors are FP32 in both.
enum class Format : int32_t { kFloat32 = 0, kQ8_0 = 1 };

// Q8_0: 32 weights share one FP32 scale, stored ahead of them so a row reads as
// one sequential stream. The last block of a row is zero-padded.
inline constexpr int kQuantGroup = 32;
struct Q8Block {
  float scale;
  int8_t values[kQuantGroup];
};
static_assert(sizeof(Q8Block) == 36);

constexpr int q8_blocks(int columns) { return (columns + kQuantGroup - 1) / kQuantGroup; }

constexpr size_t row_bytes(Format format, int columns) {
  return format == Format::kQ8_0 ? q8_blocks(columns) * sizeof(Q8Block) : columns * sizeof(float);
}

struct Config {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  float rope_theta = 10000.0f;
  Format format = Format::kFloat32;

  int head_size() const { return dim / n_heads; }
  int kv_dim() const { return head_size() * n_kv_heads; }
  size_t cache_elements(int context) const;
};

// A read-only view, not another allocation. Row r starts at
// data + r * row_bytes(format, columns).
struct Matrix {
  const std::byte* data = nullptr;
  int rows = 0;
  int columns = 0;
  Format format = Format::kFloat32;
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

// Owns one resident checkpoint. All views borrow from its data_ buffer.
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
  size_t weight_bytes() const { return data_.size(); }

 private:
  // Validate total size before allocating, then bind the views after reading.
  size_t weight_layout(bool shared, bool legacy);
  std::vector<std::byte> data_;
};

}  // namespace unremarkable
#endif
