#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "kernels.h"

// Q8_0 checks: the integer dot is exact, so the only error against an FP64
// oracle is the quantization itself, and that error is bounded by the group
// scale rather than by the magnitude of the whole row.
template <typename Random>
bool quantized_matches_reference(Random&& random) {
  using unremarkable::kQuantGroup;
  using unremarkable::q8_blocks;
  using unremarkable::Q8Block;
  const int widths[] = {1, 2, 31, 32, 33, 63, 64, 65, 96, 576, 1536};
  const int heights[] = {1, 3, 7};
  for (int columns : widths) {
    for (int rows : heights) {
      const int blocks = q8_blocks(columns);
      std::vector<float> input(columns), weights((size_t)rows * columns);
      for (float& value : input) {
        value = random();
      }
      for (float& value : weights) {
        value = random();
      }
      std::vector<Q8Block> quantized_input(blocks);
      std::vector<Q8Block> quantized_weights((size_t)rows * blocks);
      unremarkable::quantize_q8(input.data(), columns, quantized_input.data());
      for (int i = 0; i < rows; i++) {
        unremarkable::quantize_q8(weights.data() + (size_t)i * columns, columns,
                                  quantized_weights.data() + (size_t)i * blocks);
      }

      // Padding must be zero, or a tail group would contribute noise.
      for (int b = 0; b < blocks; b++) {
        for (int i = columns - b * kQuantGroup; i < kQuantGroup; i++) {
          if (i >= 0 && quantized_input[b].values[i] != 0) {
            std::fprintf(stderr, "q8 tail not zeroed: columns=%d block=%d\n", columns, b);
            return false;
          }
        }
      }

      std::vector<float> output(rows, 12345.0f);
      unremarkable::matvec_q8(quantized_input.data(), quantized_weights.data(), columns, rows,
                              output.data());
      for (int i = 0; i < rows; i++) {
        // Oracle over the values the kernel actually sees, in FP64.
        double expected = 0, magnitude = 0;
        for (int b = 0; b < blocks; b++) {
          const Q8Block& w = quantized_weights[(size_t)i * blocks + b];
          const Q8Block& x = quantized_input[b];
          long dot = 0;
          for (int k = 0; k < kQuantGroup; k++) {
            dot += w.values[k] * x.values[k];
          }
          expected += static_cast<double>(dot) * w.scale * x.scale;
          magnitude += std::abs(static_cast<double>(dot)) * w.scale * x.scale;
        }
        if (!std::isfinite(output[i]) || std::abs(output[i] - expected) > 1e-6 + 2e-6 * magnitude) {
          std::fprintf(stderr, "q8 mismatch: rows=%d columns=%d row=%d got=%g want=%g\n", rows,
                       columns, i, static_cast<double>(output[i]), expected);
          return false;
        }
      }
    }
  }
  return true;
}

// Exercise every representable weight against every supported activation,
// repeated across a group. Unit scales make overflow visible as an exact error.
bool quantized_extremes() {
  using unremarkable::Q8Block;
  Q8Block input{}, weight{};
  input.scale = weight.scale = 1;
  for (int x = -127; x <= 127; ++x) {
    for (int w = -128; w <= 127; ++w) {
      for (int k = 0; k < unremarkable::kQuantGroup; ++k) {
        input.values[k] = static_cast<int8_t>(x);
        weight.values[k] = static_cast<int8_t>(w);
      }
      float output;
      unremarkable::matvec_q8(&input, &weight, unremarkable::kQuantGroup, 1, &output);
      if (output != static_cast<float>(unremarkable::kQuantGroup * x * w)) {
        std::fprintf(stderr, "q8 integer overflow: input=%d weight=%d got=%g\n", x, w, output);
        return false;
      }
    }
  }
  return true;
}

int main() {
  uint32_t seed = 1;
  auto random = [&] {
    seed = 1664525u * seed + 1013904223u;
    return (static_cast<int>(seed >> 16) - 32768) / 32768.0f;
  };
  const int widths[] = {0, 1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 172, 576, 1536};
  const int heights[] = {1, 3, 7};
  for (int columns : widths) {
    for (int rows : heights) {
      // Offset by one float to exercise loads without 16-byte alignment.
      std::vector<float> input(columns + 2), weights(rows * columns + 2);
      std::vector<float> output(rows + 2, 12345.0f);
      for (float& value : input) {
        value = random();
      }
      for (float& value : weights) {
        value = random();
      }
      unremarkable::matvec(input.data() + 1, weights.data() + 1, columns, rows, output.data() + 1);
      for (int i = 0; i < rows; i++) {
        double expected = 0, magnitude = 0;
        for (int j = 0; j < columns; j++) {
          double term = static_cast<double>(weights[1 + i * columns + j]) * input[1 + j];
          expected += term;
          magnitude += std::abs(term);
        }
        if (!std::isfinite(output[i + 1]) ||
            std::abs(output[i + 1] - expected) > 1e-6 + 2e-6 * magnitude) {
          std::fprintf(stderr, "matvec mismatch: rows=%d columns=%d row=%d\n", rows, columns, i);
          return 1;
        }
      }
      if (output.front() != 12345.0f || output.back() != 12345.0f) {
        return 1;
      }
    }
  }
  if (!quantized_matches_reference(random) || !quantized_extremes()) {
    return 1;
  }
  std::puts("matvec: reference, tails, unaligned buffers, and Q8_0 passed");
}
