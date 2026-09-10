// FP32 Llama forward pass adapted from karpathy/llama2.c. See THIRD_PARTY.md.
#include "engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "kernels.h"

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
constexpr int32_t kVersion = 1;          // FP32 payload.
constexpr int32_t kVersionQuant = 2;     // Adds the quantization field.

}  // namespace

// First walk validates the layout without allocating; second walk binds views
// into data_. Both use the original context stored in the checkpoint header.
size_t Engine::row_bytes(int columns) const {
  if (config_.quantization == Quantization::kQ8_0) {
    return product(q8_blocks(columns), sizeof(Q8Block));
  }
  return product(columns, sizeof(float));
}

size_t Engine::weight_layout(bool shared, bool legacy) {
  size_t offset = 0;
  const size_t dim = config_.dim, layers = config_.n_layers;
  const size_t head = dim / config_.n_heads;
  const size_t kv = head * config_.n_kv_heads;
  constexpr size_t kMaximum = std::numeric_limits<ptrdiff_t>::max();
  auto advance = [&](size_t bytes) -> const std::byte* {
    if (bytes > kMaximum || offset > kMaximum - bytes) {
      throw std::runtime_error("checkpoint is too large");
    }
    const std::byte* view = data_.empty() ? nullptr : data_.data() + offset;
    offset += bytes;
    return view;
  };
  // Norm vectors stay FP32 in every format; matrices are laid out a row at a time.
  auto norm = [&](size_t count) -> const float* {
    return reinterpret_cast<const float*>(advance(product(count, sizeof(float))));
  };
  auto matrix = [&](size_t rows, size_t columns) -> const std::byte* {
    return advance(product(rows, row_bytes(static_cast<int>(columns))));
  };
  weights_.token_embedding_table = matrix(config_.vocab_size, dim);
  weights_.rms_att_weight = norm(product(layers, dim));
  weights_.wq = matrix(product(layers, dim), dim);
  weights_.wk = matrix(product(layers, kv), dim);
  weights_.wv = matrix(product(layers, kv), dim);
  weights_.wo = matrix(product(layers, dim), dim);
  weights_.rms_ffn_weight = norm(product(layers, dim));
  weights_.w1 = matrix(product(layers, config_.hidden_dim), dim);
  weights_.w2 = matrix(product(layers, dim), config_.hidden_dim);
  weights_.w3 = matrix(product(layers, config_.hidden_dim), dim);
  weights_.rms_final_weight = norm(dim);
  if (legacy) {
    norm(product(config_.seq_len, head));  // Skip both unused legacy RoPE tables.
  }
  weights_.wcls = shared ? weights_.token_embedding_table : matrix(config_.vocab_size, dim);
  return offset;
}

void Engine::project(const float* input, const std::byte* base, int columns, int rows,
                     float* output) {
  // Rows below this are not worth waking a second core for: the projections are
  // issued a few hundred times per token, and the handoff costs more than the
  // work saved on a short one.
  constexpr int kParallelRows = 64;
  if (config_.quantization == Quantization::kQ8_0) {
    quantize_q8(input, columns, state_.xq.data());
    const Q8Block* weight = reinterpret_cast<const Q8Block*>(base);
    const Q8Block* activations = state_.xq.data();
    const size_t stride = q8_blocks(columns);
    pool_.run(rows, kParallelRows, [&](int begin, int end) {
      matvec_q8(activations, weight + static_cast<size_t>(begin) * stride, columns, end - begin,
                output + begin);
    });
  } else {
    const float* weight = reinterpret_cast<const float*>(base);
    pool_.run(rows, kParallelRows, [&](int begin, int end) {
      matvec(input, weight + static_cast<size_t>(begin) * columns, columns, end - begin,
             output + begin);
    });
  }
}

Engine::Engine(const std::string& checkpoint, int context, int threads) : pool_(threads) {
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
  int32_t version = 0;
  read_exact(file, &magic, sizeof(magic));
  const bool legacy = magic != kMagic;
  size_t header_bytes = 28;
  if (legacy) {
    file.seekg(0);
  } else {
    read_exact(file, &version, sizeof(version));
    if (version != kVersion && version != kVersionQuant) {
      throw std::runtime_error("unsupported checkpoint version");
    }
    header_bytes = version == kVersionQuant ? 44 : 40;
    if (file_size < static_cast<std::streamoff>(header_bytes)) {
      throw std::runtime_error("checkpoint header is truncated");
    }
  }
  int32_t header[7];
  read_exact(file, header, sizeof(header));
  config_ =
      Config{header[0], header[1], header[2], header[3], header[4], header[5], header[6], 10000.0f};
  if (!legacy) {
    read_exact(file, &config_.rope_theta, sizeof(config_.rope_theta));
    if (!std::isfinite(config_.rope_theta) || config_.rope_theta <= 1.0f) {
      throw std::runtime_error("invalid RoPE base in checkpoint header");
    }
  }
  if (version == kVersionQuant) {
    int32_t quantization = 0;
    read_exact(file, &quantization, sizeof(quantization));
    if (quantization != static_cast<int32_t>(Quantization::kFloat32) &&
        quantization != static_cast<int32_t>(Quantization::kQ8_0)) {
      throw std::runtime_error("unsupported quantization in checkpoint header");
    }
    config_.quantization = static_cast<Quantization>(quantization);
  }
  if (config_.vocab_size == std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("invalid vocabulary size");
  }
  const bool shared = config_.vocab_size > 0;
  config_.vocab_size = std::abs(config_.vocab_size);
  if (config_.dim <= 0 || config_.hidden_dim <= 0 || config_.n_layers <= 0 ||
      config_.n_heads <= 0 || config_.n_kv_heads <= 0 || config_.vocab_size < 259 ||
      config_.seq_len <= 0 || config_.dim % config_.n_heads ||
      config_.n_heads % config_.n_kv_heads || (config_.dim / config_.n_heads) % 2) {
    throw std::runtime_error("invalid model header; expected an FP32 Llama checkpoint");
  }
  const size_t bytes = weight_layout(shared, legacy);
  if (static_cast<uint64_t>(file_size) - header_bytes != bytes) {
    throw std::runtime_error("checkpoint size does not match its header");
  }
  if (context < 0 || context > config_.seq_len) {
    throw std::runtime_error("requested context exceeds model context");
  }
  data_.resize(bytes);
  read_exact(file, data_.data(), bytes);
  weight_layout(shared, legacy);
  if (context) {
    config_.seq_len = context;
  }

  const size_t kv_dim = (config_.dim / config_.n_heads) * config_.n_kv_heads;
  const size_t cache = product(product(config_.n_layers, config_.seq_len), kv_dim);
  product(cache, 2 * sizeof(float));
  state_.x.resize(config_.dim);
  state_.xb.resize(config_.dim);
  state_.xb2.resize(config_.dim);
  state_.hb.resize(config_.hidden_dim);
  state_.hb2.resize(config_.hidden_dim);
  state_.q.resize(config_.dim);
  if (config_.quantization == Quantization::kQ8_0) {
    state_.xq.resize(q8_blocks(std::max(config_.dim, config_.hidden_dim)));
  }
  state_.att.resize(product(config_.n_heads, config_.seq_len));
  state_.logits.resize(config_.vocab_size);
  state_.key_cache.resize(cache);
  state_.value_cache.resize(cache);
}

size_t Engine::kv_bytes() const {
  return (state_.key_cache.size() + state_.value_cache.size()) * sizeof(float);
}

void Engine::reset() {
  // Forward overwrites every cache position before attention is allowed to read
  // it, so starting a new sequence does not require zeroing the entire cache.
  next_position_ = 0;
}

std::span<float> Engine::forward(int token, int position) {
  if (token < 0 || token >= config_.vocab_size) {
    throw std::runtime_error("token ID out of range");
  }
  if (position != next_position_ || position < 0 || position >= config_.seq_len) {
    throw std::runtime_error("forward position must follow the cached prefix and fit context");
  }
  const auto& config = config_;
  const auto& weights = weights_;
  auto& state = state_;
  const int dim = config.dim;
  const int hidden_dim = config.hidden_dim;
  const int head_size = dim / config.n_heads;
  const int kv_dim = head_size * config.n_kv_heads;
  float* residual = state.x.data();
  // A quantized row carries its scales inline, so its stride is not the column count.
  const size_t row_dim = row_bytes(dim);
  const size_t row_hidden = row_bytes(hidden_dim);

  if (config.quantization == Quantization::kQ8_0) {
    dequantize_q8(reinterpret_cast<const Q8Block*>(weights.token_embedding_table) +
                      static_cast<size_t>(token) * q8_blocks(dim),
                  dim, residual);
  } else {
    embedding_lookup(reinterpret_cast<const float*>(weights.token_embedding_table), token, dim,
                     residual);
  }

  for (size_t layer = 0; layer < static_cast<size_t>(config.n_layers); layer++) {
    // Each layer owns a cache slice; projections write the current slot
    // directly.
    const size_t cache_offset = layer * config.seq_len * kv_dim;
    float* key_cache = state.key_cache.data() + cache_offset;
    float* value_cache = state.value_cache.data() + cache_offset;
    float* key = key_cache + static_cast<size_t>(position) * kv_dim;
    float* value = value_cache + static_cast<size_t>(position) * kv_dim;

    // Normalize, project Q/K/V, apply RoPE, and attend to the cached prefix.
    rmsnorm(residual, weights.rms_att_weight + layer * dim, dim, state.xb.data());
    project(state.xb.data(), weights.wq + layer * dim * row_dim, dim, dim, state.q.data());
    project(state.xb.data(), weights.wk + layer * kv_dim * row_dim, dim, kv_dim, key);
    project(state.xb.data(), weights.wv + layer * kv_dim * row_dim, dim, kv_dim, value);
    rope_inplace(position, head_size, dim, kv_dim, config.rope_theta, state.q.data(), key);
    causal_attention(state.q.data(), key_cache, value_cache, config.n_heads, config.n_kv_heads,
                     head_size, config.seq_len, position, state.att.data(), state.xb.data());
    project(state.xb.data(), weights.wo + layer * dim * row_dim, dim, dim, state.xb2.data());
    add_inplace(state.xb2.data(), dim, residual);

    // Feed-forward: down(silu(gate(x)) * up(x)), then residual addition.
    rmsnorm(residual, weights.rms_ffn_weight + layer * dim, dim, state.xb.data());
    project(state.xb.data(), weights.w1 + layer * hidden_dim * row_dim, dim, hidden_dim,
            state.hb.data());
    project(state.xb.data(), weights.w3 + layer * hidden_dim * row_dim, dim, hidden_dim,
            state.hb2.data());
    swiglu_inplace(state.hb2.data(), hidden_dim, state.hb.data());
    project(state.hb.data(), weights.w2 + layer * dim * row_hidden, hidden_dim, dim,
            state.xb.data());
    add_inplace(state.xb.data(), dim, residual);
  }

  // Normalize the final residual stream and score every vocabulary token.
  rmsnorm(residual, weights.rms_final_weight, dim, residual);
  project(residual, weights.wcls, dim, config.vocab_size, state.logits.data());
  next_position_++;
  return state.logits;
}

Sampler::Sampler(int vocabulary, float temperature, float top_p, uint64_t seed)
    : temperature_(temperature), top_p_(top_p), rng_state_(seed) {
  if (vocabulary < 2 || !std::isfinite(temperature) || temperature < 0 || !std::isfinite(top_p) ||
      top_p < 0 || top_p > 1 || seed == 0) {
    throw std::runtime_error("invalid sampling options");
  }
  candidates_.resize(vocabulary);
}

float Sampler::random_float() {
  // xorshift64*, matching the reference's random stream.
  rng_state_ ^= rng_state_ >> 12;
  rng_state_ ^= rng_state_ << 25;
  rng_state_ ^= rng_state_ >> 27;
  uint32_t value = (rng_state_ * 0x2545F4914F6CDD1DULL) >> 32;
  return (value >> 8) / 16777216.0f;
}

int Sampler::sample(std::span<float> logits) {
  if (logits.size() != candidates_.size()) {
    throw std::runtime_error("sampler vocabulary size mismatch");
  }
  for (float value : logits) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite model logits");
    }
  }
  if (temperature_ == 0) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  for (float& value : logits) {
    value /= temperature_;
    if (!std::isfinite(value)) {
      throw std::runtime_error("temperature scaling overflowed");
    }
  }
  softmax_inplace(logits.data(), static_cast<int>(logits.size()));
  const float coin = random_float();
  if (top_p_ == 0 || top_p_ == 1) {
    float cumulative = 0;
    for (size_t i = 0; i < logits.size(); i++) {
      cumulative += logits[i];
      if (coin < cumulative) {
        return static_cast<int>(i);
      }
    }
    return static_cast<int>(logits.size() - 1);
  }

  size_t count = 0;
  const float cutoff = (1.0f - top_p_) / (logits.size() - 1);
  for (size_t i = 0; i < logits.size(); i++) {
    if (logits[i] >= cutoff) {
      candidates_[count++] = {logits[i], static_cast<int>(i)};
    }
  }
  if (!count) {
    throw std::runtime_error("empty sampling distribution");
  }
  std::sort(
      candidates_.begin(), candidates_.begin() + count, [](const Candidate& a, const Candidate& b) {
        return a.probability != b.probability ? a.probability > b.probability : a.token < b.token;
      });
  float cumulative = 0;
  size_t kept = 0;
  do {
    cumulative += candidates_[kept++].probability;
  } while (kept < count && cumulative <= top_p_);
  const float target = coin * cumulative;
  cumulative = 0;
  for (size_t i = 0; i < kept; i++) {
    cumulative += candidates_[i].probability;
    if (target < cumulative) {
      return candidates_[i].token;
    }
  }
  return candidates_[kept - 1].token;
}

}  // namespace unremarkable
