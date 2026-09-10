// Adapted from karpathy/llama2.c; see THIRD_PARTY.md and LICENSE.
#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
#include <arm_neon.h>
#endif

namespace unremarkable {
namespace {

// int8 products fit int16, and vpadal widens to int32 before eight can overflow.
inline int32_t dot_group(const int8_t* a, const int8_t* b) {
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
  const int8x16_t a0 = vld1q_s8(a);
  const int8x16_t a1 = vld1q_s8(a + 16);
  const int8x16_t b0 = vld1q_s8(b);
  const int8x16_t b1 = vld1q_s8(b + 16);
  int32x4_t acc = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(a0), vget_low_s8(b0)));
  acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(a0), vget_high_s8(b0)));
  acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(a1), vget_low_s8(b1)));
  acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(a1), vget_high_s8(b1)));
  const int32x2_t pair = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  return vget_lane_s32(vpadd_s32(pair, pair), 0);
#else
  int32_t sum = 0;
  for (int i = 0; i < kQuantGroup; i++) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

}  // namespace

void quantize_q8(const float* input, int size, Q8Block* output) {
  const int blocks = q8_blocks(size);
  for (int b = 0; b < blocks; b++) {
    const int base = b * kQuantGroup;
    const int count = std::min(kQuantGroup, size - base);
    float largest = 0;
    for (int i = 0; i < count; i++) {
      largest = std::fmax(largest, std::fabs(input[base + i]));
    }
    Q8Block& block = output[b];
    block.scale = largest / 127.0f;
    // A zero group would divide by zero; its values are already exactly zero.
    const float inverse = largest > 0 ? 127.0f / largest : 0.0f;
    for (int i = 0; i < count; i++) {
      const long rounded = std::lround(input[base + i] * inverse);
      block.values[i] = static_cast<int8_t>(std::clamp(rounded, -127L, 127L));
    }
    for (int i = count; i < kQuantGroup; i++) {
      block.values[i] = 0;
    }
  }
}

void dequantize_q8(const Q8Block* input, int size, float* output) {
  for (int i = 0; i < size; i++) {
    const Q8Block& block = input[i / kQuantGroup];
    output[i] = block.scale * block.values[i % kQuantGroup];
  }
}

void matvec_q8(const Q8Block* input, const Q8Block* weight, int columns, int rows, float* output) {
  const int blocks = q8_blocks(columns);
  for (int i = 0; i < rows; i++) {
    const Q8Block* row = weight + static_cast<size_t>(i) * blocks;
    float sum = 0;
    for (int b = 0; b < blocks; b++) {
      // Prefetching, not width, is what this loop gains from. PLD never faults,
      // so running past the last row is harmless.
      __builtin_prefetch(reinterpret_cast<const char*>(row + b) + 256);
      const int32_t dot = dot_group(row[b].values, input[b].values);
      sum += static_cast<float>(dot) * (row[b].scale * input[b].scale);
    }
    output[i] = sum;
  }
}

void rmsnorm(const float* input, const float* weight, int size, float* output) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += input[i] * input[i];
  }
  sum /= size;
  sum += 1e-5f;
  const float scale = 1.0f / std::sqrt(sum);
  for (int i = 0; i < size; i++) {
    output[i] = weight[i] * (scale * input[i]);
  }
}

void softmax_inplace(float* values, int size) {
  float maximum = values[0];
  for (int i = 1; i < size; i++) {
    if (values[i] > maximum) {
      maximum = values[i];
    }
  }
  float sum = 0;
  for (int i = 0; i < size; i++) {
    values[i] = std::exp(values[i] - maximum);
    sum += values[i];
  }
  for (int i = 0; i < size; i++) {
    values[i] /= sum;
  }
}

void matvec(const float* input, const float* weight, int columns, int rows, float* output) {
  for (int i = 0; i < rows; i++) {
    const float* row = weight + static_cast<size_t>(i) * columns;
    float sum = 0;
    int j = 0;
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
    // Four independent vector accumulators (16 partial sums) let successive
    // operations overlap. Weights stay in their original row-major layout.
    float32x4_t sums0 = vdupq_n_f32(0);
    float32x4_t sums1 = vdupq_n_f32(0);
    float32x4_t sums2 = vdupq_n_f32(0);
    float32x4_t sums3 = vdupq_n_f32(0);
    for (; j <= columns - 16; j += 16) {
      sums0 = vaddq_f32(sums0, vmulq_f32(vld1q_f32(row + j), vld1q_f32(input + j)));
      sums1 = vaddq_f32(sums1, vmulq_f32(vld1q_f32(row + j + 4), vld1q_f32(input + j + 4)));
      sums2 = vaddq_f32(sums2, vmulq_f32(vld1q_f32(row + j + 8), vld1q_f32(input + j + 8)));
      sums3 = vaddq_f32(sums3, vmulq_f32(vld1q_f32(row + j + 12), vld1q_f32(input + j + 12)));
    }
    float32x4_t sums = vaddq_f32(vaddq_f32(sums0, sums1), vaddq_f32(sums2, sums3));
    for (; j <= columns - 4; j += 4) {
      sums = vaddq_f32(sums, vmulq_f32(vld1q_f32(row + j), vld1q_f32(input + j)));
    }
    float32x2_t halves = vadd_f32(vget_low_f32(sums), vget_high_f32(sums));
    sum = vget_lane_f32(vpadd_f32(halves, halves), 0);
#endif
    for (; j < columns; j++) {
      sum += row[j] * input[j];
    }
    output[i] = sum;
  }
}

void embedding_lookup(const float* table, int token, int dim, float* output) {
  const float* row = table + static_cast<size_t>(token) * dim;
  std::memcpy(output, row, dim * sizeof(float));
}

void rope_inplace(int position, int head_size, int query_size, int key_size, float theta,
                  float* query, float* key) {
  for (int i = 0; i < query_size; i += 2) {
    int head_dim = i % head_size;
    float frequency = 1.0f / std::pow(theta, head_dim / static_cast<float>(head_size));
    float angle = position * frequency;
    float cosine = std::cos(angle);
    float sine = std::sin(angle);
    // Query and key share the rotation wherever both have a head component.
    int vectors = i < key_size ? 2 : 1;
    for (int v = 0; v < vectors; v++) {
      float* vector = v == 0 ? query : key;
      float v0 = vector[i];
      float v1 = vector[i + 1];
      vector[i] = v0 * cosine - v1 * sine;
      vector[i + 1] = v0 * sine + v1 * cosine;
    }
  }
}

void causal_attention(const float* query, const float* key_cache, const float* value_cache,
                      int n_heads, int n_kv_heads, int head_size, int context, int position,
                      float* scores, float* output) {
  const int kv_dim = n_kv_heads * head_size;
  const int queries_per_kv_head = n_heads / n_kv_heads;
  for (int h = 0; h < n_heads; h++) {
    const float* q = query + h * head_size;
    float* head_scores = scores + static_cast<size_t>(h) * context;
    const int kv_head_offset = (h / queries_per_kv_head) * head_size;
    for (int t = 0; t <= position; t++) {
      const float* k = key_cache + static_cast<size_t>(t) * kv_dim + kv_head_offset;
      float score = 0.0f;
      for (int i = 0; i < head_size; i++) {
        score += q[i] * k[i];
      }
      score /= std::sqrt(static_cast<float>(head_size));
      head_scores[t] = score;
    }

    softmax_inplace(head_scores, position + 1);

    float* head_output = output + h * head_size;
    std::memset(head_output, 0, head_size * sizeof(float));
    for (int t = 0; t <= position; t++) {
      const float* v = value_cache + static_cast<size_t>(t) * kv_dim + kv_head_offset;
      float probability = head_scores[t];
      for (int i = 0; i < head_size; i++) {
        head_output[i] += probability * v[i];
      }
    }
  }
}

void swiglu_inplace(const float* up, int size, float* gate) {
  for (int i = 0; i < size; i++) {
    float value = gate[i];
    value *= (1.0f / (1.0f + std::exp(-value)));
    value *= up[i];
    gate[i] = value;
  }
}

void add_inplace(const float* input, int size, float* output) {
  for (int i = 0; i < size; i++) {
    output[i] += input[i];
  }
}

}  // namespace unremarkable
