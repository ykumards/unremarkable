#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "kernels.h"

using unremarkable::causal_attention;

// Compare against scalar attention, including grouped heads and SIMD tails.
int main() {
  constexpr int context = 65;
  constexpr float untouched = -999;
  for (int head_size : {1, 3, 4, 7, 16, 64}) {
    for (int kv_heads : {1, 3, 9}) {
      constexpr int heads = 9;
      const int kv_dim = kv_heads * head_size;
      for (int position : {0, 1, 31, 64}) {
        std::vector<float> query(heads * head_size);
        std::vector<float> keys(context * kv_dim, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> values = keys;
        for (size_t i = 0; i < query.size(); ++i) {
          query[i] = std::sin(static_cast<float>(i) * 0.31f);
        }
        for (int i = 0; i < (position + 1) * kv_dim; ++i) {
          keys[i] = std::cos(i * 0.17f);
          values[i] = std::sin(i * 0.23f);
        }
        std::vector<float> scores(heads * context, untouched);
        std::vector<float> output(heads * head_size + 1, untouched);
        // Split through a KV group to check global head-to-cache indexing.
        for (const auto [begin, end] : {std::pair{0, 4}, std::pair{4, heads}}) {
          causal_attention(query.data(), keys.data(), values.data(), heads, kv_heads, head_size,
                           context, position, begin, end, scores.data(), output.data());
        }
        for (int h = 0; h < heads; ++h) {
          std::vector<float> probabilities(position + 1);
          const int kv_offset = (h / (heads / kv_heads)) * head_size;
          for (int t = 0; t <= position; ++t) {
            float dot = 0;
            for (int i = 0; i < head_size; ++i) {
              dot += query[h * head_size + i] * keys[t * kv_dim + kv_offset + i];
            }
            probabilities[t] = dot / std::sqrt(static_cast<float>(head_size));
          }
          const float largest = *std::max_element(probabilities.begin(), probabilities.end());
          float denominator = 0;
          for (float& p : probabilities) {
            p = std::exp(p - largest);
            denominator += p;
          }
          for (float& p : probabilities) {
            p /= denominator;
          }
          for (int i = 0; i < head_size; ++i) {
            float expected = 0;
            for (int t = 0; t <= position; ++t) {
              expected += probabilities[t] * values[t * kv_dim + kv_offset + i];
            }
            const float actual = output[h * head_size + i];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 2e-6f) {
              std::fprintf(stderr, "attention mismatch: size=%d kv=%d pos=%d head=%d i=%d\n",
                           head_size, kv_heads, position, h, i);
              return 1;
            }
          }
          for (int t = position + 1; t < context; ++t) {
            if (scores[h * context + t] != untouched) {
              return 1;
            }
          }
        }
        if (output.back() != untouched) {
          return 1;
        }
      }
    }
  }
  std::puts("attention: scalar reference, grouped heads, tails, and causal boundaries passed");
}
