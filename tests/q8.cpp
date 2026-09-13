#include <algorithm>
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

int main() {
  if (!extremes() || !mixed_rows()) {
    return 1;
  }
  std::puts("Q8: all 65280 input/weight pairs, mixed lanes, scales, zero input, and tails passed");
}
