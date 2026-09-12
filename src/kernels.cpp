// Adapted from karpathy/llama2.c; see THIRD_PARTY.md and LICENSE.
#include "kernels.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
#include <arm_neon.h>
#endif

#include "model.h"

namespace unremarkable {

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

// Each output is one row's dot product. Each weight cache line is prefetched
// 256 bytes ahead so the loop does not stall on every miss.
void matvec(const float* input, Matrix weight, float* output) {
  constexpr int floats_per_line = 16;
  constexpr int prefetch_ahead = 64;
  for (int row = 0; row < weight.rows; ++row) {
    const float* weights = weight.data + static_cast<size_t>(row) * weight.columns;
    float sum = 0;
    int column = 0;
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
    // 16 independent partial sums; this reorders the additions.
    float32x4_t sums0 = vdupq_n_f32(0);
    float32x4_t sums1 = vdupq_n_f32(0);
    float32x4_t sums2 = vdupq_n_f32(0);
    float32x4_t sums3 = vdupq_n_f32(0);
    for (; column <= weight.columns - floats_per_line; column += floats_per_line) {
      __builtin_prefetch(weights + column + prefetch_ahead);
      const float* w = weights + column;
      const float* x = input + column;
      sums0 = vaddq_f32(sums0, vmulq_f32(vld1q_f32(w), vld1q_f32(x)));
      sums1 = vaddq_f32(sums1, vmulq_f32(vld1q_f32(w + 4), vld1q_f32(x + 4)));
      sums2 = vaddq_f32(sums2, vmulq_f32(vld1q_f32(w + 8), vld1q_f32(x + 8)));
      sums3 = vaddq_f32(sums3, vmulq_f32(vld1q_f32(w + 12), vld1q_f32(x + 12)));
    }
    float32x4_t sums = vaddq_f32(vaddq_f32(sums0, sums1), vaddq_f32(sums2, sums3));
    for (; column <= weight.columns - 4; column += 4) {
      sums = vaddq_f32(sums, vmulq_f32(vld1q_f32(weights + column), vld1q_f32(input + column)));
    }
    float32x2_t halves = vadd_f32(vget_low_f32(sums), vget_high_f32(sums));
    sum = vget_lane_f32(vpadd_f32(halves, halves), 0);
#else
    while (column <= weight.columns - floats_per_line) {
      __builtin_prefetch(weights + column + prefetch_ahead);
      for (const int end = column + floats_per_line; column < end; ++column) {
        sum += weights[column] * input[column];
      }
    }
#endif
    for (; column < weight.columns; ++column) {
      sum += weights[column] * input[column];
    }
    output[row] = sum;
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
