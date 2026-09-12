// FP32 checkpoint loading adapted from karpathy/llama2.c. See THIRD_PARTY.md.
#include "model.h"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace unremarkable {
namespace {

size_t product(size_t a, size_t b) {
  constexpr size_t kMaximum = std::numeric_limits<ptrdiff_t>::max();
  if (b && a > kMaximum / b) {
    throw std::runtime_error("model dimensions overflow address space");
  }
  return a * b;
}

void read_exact(std::ifstream& file, void* output, size_t bytes) {
  if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) ||
      !file.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("truncated or unreadable checkpoint");
  }
}

constexpr uint32_t kMagic = 0x4B524E55;  // "UNRK", absent from legacy checkpoints.
constexpr int32_t kVersion = 1;

}  // namespace

size_t Model::weight_layout(bool shared, bool legacy) {
  size_t offset = 0;
  const size_t dim = config.dim, layer_count = config.n_layers;
  const size_t head = dim / config.n_heads;
  const size_t kv = head * config.n_kv_heads;
  auto tensor = [&](size_t a, size_t b, size_t c = 1) -> const float* {
    const size_t count = product(product(a, b), c);
    constexpr size_t kMaximum = std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
    if (count > kMaximum || offset > kMaximum - count) {
      throw std::runtime_error("checkpoint is too large");
    }
    const float* view = data_.empty() ? nullptr : data_.data() + offset;
    offset += count;
    return view;
  };
  const float* token_embedding_table = tensor(config.vocab_size, dim);
  const float* rms_att_weight = tensor(layer_count, dim);
  const float* wq = tensor(layer_count, dim, dim);
  const float* wk = tensor(layer_count, kv, dim);
  const float* wv = tensor(layer_count, kv, dim);
  const float* wo = tensor(layer_count, dim, dim);
  const float* rms_ffn_weight = tensor(layer_count, dim);
  const float* w1 = tensor(layer_count, config.hidden_dim, dim);
  const float* w2 = tensor(layer_count, dim, config.hidden_dim);
  const float* w3 = tensor(layer_count, config.hidden_dim, dim);
  const float* rms_final_weight = tensor(dim, 1);
  if (legacy) {
    tensor(config.seq_len, head);  // Skip both unused legacy RoPE tables.
  }
  const float* wcls = shared ? token_embedding_table : tensor(config.vocab_size, dim);
  // Only construct views once the full payload has been validated and loaded.
  if (!data_.empty()) {
    embedding = {token_embedding_table, config.vocab_size, config.dim};
    classifier = {wcls, config.vocab_size, config.dim};
    final_norm = {rms_final_weight, dim};
    layers.resize(layer_count);
    for (size_t layer = 0; layer < layer_count; ++layer) {
      LayerWeights& weights = layers[layer];
      weights.attention_norm = {rms_att_weight + layer * dim, dim};
      weights.feed_forward_norm = {rms_ffn_weight + layer * dim, dim};
      weights.query = {wq + layer * dim * dim, config.dim, config.dim};
      weights.key = {wk + layer * kv * dim, static_cast<int>(kv), config.dim};
      weights.value = {wv + layer * kv * dim, static_cast<int>(kv), config.dim};
      weights.attention_output = {wo + layer * dim * dim, config.dim, config.dim};
      weights.gate = {w1 + layer * config.hidden_dim * dim, config.hidden_dim, config.dim};
      weights.up = {w3 + layer * config.hidden_dim * dim, config.hidden_dim, config.dim};
      weights.down = {w2 + layer * dim * config.hidden_dim, config.dim, config.hidden_dim};
    }
  }
  return product(offset, sizeof(float));
}

Model::Model(const std::string& checkpoint) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::ifstream file(checkpoint, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("cannot open checkpoint: " + checkpoint);
  }
  const auto file_size = file.tellg();
  if (file_size < 28) {
    throw std::runtime_error("checkpoint header is truncated");
  }
  // A tagged checkpoint leads with the magic and carries its own RoPE base; a
  // legacy llama2.c checkpoint starts straight into the seven header integers.
  file.seekg(0);
  uint32_t magic = 0;
  read_exact(file, &magic, sizeof(magic));
  const bool legacy = magic != kMagic;
  size_t header_bytes = 28;
  if (legacy) {
    file.seekg(0);
  } else {
    int32_t version = 0;
    read_exact(file, &version, sizeof(version));
    if (version != kVersion) {
      throw std::runtime_error("unsupported checkpoint version");
    }
    header_bytes = 40;
    if (file_size < static_cast<std::streamoff>(header_bytes)) {
      throw std::runtime_error("checkpoint header is truncated");
    }
  }
  int32_t header[7];
  read_exact(file, header, sizeof(header));
  config =
      Config{header[0], header[1], header[2], header[3], header[4], header[5], header[6], 10000.0f};
  if (!legacy) {
    read_exact(file, &config.rope_theta, sizeof(config.rope_theta));
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 1.0f) {
      throw std::runtime_error("invalid RoPE base in checkpoint header");
    }
  }
  if (config.vocab_size == std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("invalid vocabulary size");
  }
  const bool shared = config.vocab_size > 0;
  config.vocab_size = std::abs(config.vocab_size);
  if (config.dim <= 0 || config.hidden_dim <= 0 || config.n_layers <= 0 || config.n_heads <= 0 ||
      config.n_kv_heads <= 0 || config.vocab_size < 259 || config.seq_len <= 0 ||
      config.dim % config.n_heads || config.n_heads % config.n_kv_heads ||
      (config.dim / config.n_heads) % 2) {
    throw std::runtime_error("invalid model header; expected an FP32 Llama checkpoint");
  }
  const size_t bytes = weight_layout(shared, legacy);
  if (static_cast<uint64_t>(file_size) - header_bytes != bytes) {
    throw std::runtime_error("checkpoint size does not match its header");
  }
  data_.resize(bytes / sizeof(float));
  read_exact(file, data_.data(), bytes);
  weight_layout(shared, legacy);
}

size_t Config::cache_elements(int context) const {
  const size_t count = product(product(n_layers, context), kv_dim());
  product(count, 2 * sizeof(float));  // Keys and values together must fit.
  return count;
}

}  // namespace unremarkable
