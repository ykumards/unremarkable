#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "kernels.h"
#include "model.h"

using namespace unremarkable;

bool extremes() {
  Q8Block input{}, weight{};
  input.scale = weight.scale = 1;
  Matrix matrix{reinterpret_cast<const std::byte*>(&weight), 1, kQuantGroup, Format::kQ8_0};
  for (int x = -127; x <= 127; ++x) {
    for (int w = -128; w <= 127; ++w) {
      std::fill_n(input.values, kQuantGroup, static_cast<int8_t>(x));
      std::fill_n(weight.values, kQuantGroup, static_cast<int8_t>(w));
      float actual;
      matvec_q8(&input, matrix, &actual);
      if (actual != static_cast<float>(kQuantGroup * x * w)) {
        std::fprintf(stderr, "Q8 overflow: input=%d weight=%d output=%g\n", x, w, actual);
        return false;
      }
    }
  }
  return true;
}

bool mixed_rows() {
  uint32_t seed = 1;
  auto random = [&] {
    seed = seed * 1664525u + 1013904223u;
    return seed;
  };
  for (int columns : {1, 31, 32, 33, 63, 64, 65, 576, 1536}) {
    const int blocks = q8_blocks(columns);
    std::vector<float> input(columns);
    std::vector<Q8Block> quantized(blocks), weights(3 * blocks);
    Matrix matrix{reinterpret_cast<const std::byte*>(weights.data()), 3, columns, Format::kQ8_0};
    for (int trial = 0; trial < 10; ++trial) {
      for (float& value : input) {
        value = trial == 0 ? 0 : (static_cast<int>(random() % 20001) - 10000) / 333.0f;
      }
      quantize_q8(input.data(), columns, quantized.data());
      for (int b = 0; b < blocks; ++b) {
        for (int i = 0; i < kQuantGroup; ++i) {
          if (quantized[b].values[i] < -127 ||
              (b * kQuantGroup + i >= columns && quantized[b].values[i] != 0)) {
            return false;
          }
        }
      }
      for (Q8Block& block : weights) {
        block.scale = static_cast<float>(random() % 1000) / 997.0f;
        for (int8_t& value : block.values) {
          value = static_cast<int8_t>(static_cast<int>(random() % 256) - 128);
        }
      }
      float actual[3];
      matvec_q8(quantized.data(), matrix, actual);
      for (int row = 0; row < 3; ++row) {
        float expected = 0;
        for (int b = 0; b < blocks; ++b) {
          const auto& weight = weights[row * blocks + b];
          int32_t dot = 0;
          for (int i = 0; i < kQuantGroup; ++i) {
            dot += static_cast<int32_t>(weight.values[i]) * quantized[b].values[i];
          }
          expected += static_cast<float>(dot) * (weight.scale * quantized[b].scale);
        }
        if (actual[row] != expected) {
          std::fprintf(stderr, "Q8 mismatch: columns=%d row=%d\n", columns, row);
          return false;
        }
      }
    }
  }
  return true;
}

// quantize_q8 must match the std::lround definition exactly, including halves.
bool rounding() {
  uint32_t seed = 7;
  auto random = [&] {
    seed = seed * 1664525u + 1013904223u;
    return seed;
  };
  for (int columns : {1, 31, 32, 33, 576, 1536}) {
    const int blocks = q8_blocks(columns);
    std::vector<float> input(columns);
    std::vector<Q8Block> actual(blocks);
    for (int trial = 0; trial < 20; ++trial) {
      for (int i = 0; i < columns; ++i) {
        input[i] = (static_cast<int>(random() % 20001) - 10000) / 97.0f;
      }
      if (trial == 1) {
        // A maximum of 127 makes the inverse exactly 1, so these land on halves.
        for (int i = 0; i < columns; ++i) {
          input[i] = i % kQuantGroup == 0 ? 127.0f : (i % 254) - 126.5f;
        }
      }
      if (trial == 2) {
        for (int i = 0; i < columns; ++i) {
          const float edges[] = {127.0f, 0.49999997f, -0.49999997f, 0.5f, -0.5f, 2.5f, -2.5f};
          input[i] = edges[i % 7];
        }
      }
      quantize_q8(input.data(), columns, actual.data());
      for (int b = 0; b < blocks; ++b) {
        const int base = b * kQuantGroup;
        const int count = std::min(kQuantGroup, columns - base);
        float largest = 0;
        for (int i = 0; i < count; ++i) {
          largest = std::fmax(largest, std::fabs(input[base + i]));
        }
        const float inverse = largest > 0 ? 127.0f / largest : 0.0f;
        if (actual[b].scale != largest / 127.0f) {
          std::fprintf(stderr, "quantize scale: columns=%d block=%d\n", columns, b);
          return false;
        }
        for (int i = 0; i < kQuantGroup; ++i) {
          const long expected =
              i < count ? std::clamp(std::lround(input[base + i] * inverse), -127L, 127L) : 0;
          if (actual[b].values[i] != expected) {
            std::fprintf(stderr, "quantize value: columns=%d block=%d i=%d got=%d want=%ld\n",
                         columns, b, i, actual[b].values[i], expected);
            return false;
          }
        }
      }
    }
  }
  return true;
}

// Precomputed angles must reproduce the per-pair formula bit for bit.
bool rope_table() {
  const int head_size = 64, query_size = 576, key_size = 192;
  std::vector<float> cosines(head_size / 2), sines(head_size / 2);
  std::vector<float> query(query_size), key(key_size), expected_query, expected_key;
  for (float theta : {10000.0f, 100000.0f}) {
    for (int position = 0; position <= 600; ++position) {
      for (int i = 0; i < query_size; ++i) {
        query[i] = std::sin(0.37f * i + position);
      }
      for (int i = 0; i < key_size; ++i) {
        key[i] = std::cos(0.23f * i - position);
      }
      expected_query = query;
      expected_key = key;
      for (int i = 0; i < query_size; i += 2) {
        const int head_dim = i % head_size;
        const float frequency = 1.0f / std::pow(theta, head_dim / static_cast<float>(head_size));
        const float angle = position * frequency;
        const float cosine = std::cos(angle), sine = std::sin(angle);
        for (std::vector<float>* vector : {&expected_query, &expected_key}) {
          if (i < static_cast<int>(vector->size())) {
            const float v0 = (*vector)[i], v1 = (*vector)[i + 1];
            (*vector)[i] = v0 * cosine - v1 * sine;
            (*vector)[i + 1] = v0 * sine + v1 * cosine;
          }
        }
      }
      rope_angles(position, head_size, theta, cosines.data(), sines.data());
      rope_inplace(cosines.data(), sines.data(), head_size, query_size, key_size, query.data(),
                   key.data());
      if (query != expected_query || key != expected_key) {
        std::fprintf(stderr, "rope mismatch: theta=%g position=%d\n", theta, position);
        return false;
      }
    }
  }
  return true;
}

// SwiGLU against a double-precision reference, including saturated inputs.
bool swiglu_close() {
  std::vector<float> gate, up;
  for (int i = -2000; i <= 2000; ++i) {
    gate.push_back(i / 20.0f);
    up.push_back(1.0f + (i % 7) / 3.0f);
  }
  for (float value : {0.0f, -0.0f, 1e-30f, -1e-30f, 87.0f, -87.0f, 88.5f, -88.5f, 95.0f, -95.0f}) {
    gate.push_back(value);
    up.push_back(-1.5f);
  }
  const std::vector<float> original = gate;
  swiglu_inplace(up.data(), static_cast<int>(gate.size()), gate.data());
  for (size_t i = 0; i < gate.size(); ++i) {
    const double g = original[i];
    const double expected = g / (1.0 + std::exp(-g)) * up[i];
    // Below -87 the clamped exp leaves ~1e-37 where the true value is smaller;
    // either vanishes in the projection sums the output feeds.
    if (std::fabs(gate[i] - expected) > 2e-6 * std::fabs(expected) + 1e-30) {
      std::fprintf(stderr, "swiglu: gate=%g got=%.9g want=%.9g\n", original[i], gate[i], expected);
      return false;
    }
  }
  return true;
}

int main() {
  if (!extremes() || !mixed_rows() || !rounding() || !rope_table() || !swiglu_close()) {
    return 1;
  }
  std::puts("kernels: Q8 pairs, mixed lanes, tails, exact rounding, RoPE table, and SwiGLU passed");
}
