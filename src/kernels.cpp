// Adapted from karpathy/llama2.c; see THIRD_PARTY.md and LICENSE.
#include "kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
#include <arm_neon.h>
#endif

#include "model.h"

namespace unremarkable {
namespace {

// One FP32 row. Prefetch weights 256 bytes ahead while processing 16 floats.
float dot_fp32(const float* weights, const float* input, int size) {
  constexpr int floats_per_block = 16;
  constexpr int prefetch_ahead = 64;
  float sum = 0;
  int column = 0;
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
  // 16 independent partial sums; this reorders the additions.
  float32x4_t sums0 = vdupq_n_f32(0);
  float32x4_t sums1 = vdupq_n_f32(0);
  float32x4_t sums2 = vdupq_n_f32(0);
  float32x4_t sums3 = vdupq_n_f32(0);
  for (; column <= size - floats_per_block; column += floats_per_block) {
    __builtin_prefetch(weights + column + prefetch_ahead);
    const float* w = weights + column;
    const float* x = input + column;
    sums0 = vaddq_f32(sums0, vmulq_f32(vld1q_f32(w), vld1q_f32(x)));
    sums1 = vaddq_f32(sums1, vmulq_f32(vld1q_f32(w + 4), vld1q_f32(x + 4)));
    sums2 = vaddq_f32(sums2, vmulq_f32(vld1q_f32(w + 8), vld1q_f32(x + 8)));
    sums3 = vaddq_f32(sums3, vmulq_f32(vld1q_f32(w + 12), vld1q_f32(x + 12)));
  }
  float32x4_t sums = vaddq_f32(vaddq_f32(sums0, sums1), vaddq_f32(sums2, sums3));
  for (; column <= size - 4; column += 4) {
    sums = vaddq_f32(sums, vmulq_f32(vld1q_f32(weights + column), vld1q_f32(input + column)));
  }
  float32x2_t halves = vadd_f32(vget_low_f32(sums), vget_high_f32(sums));
  sum = vget_lane_f32(vpadd_f32(halves, halves), 0);
#else
  while (column <= size - floats_per_block) {
    __builtin_prefetch(weights + column + prefetch_ahead);
    for (const int end = column + floats_per_block; column < end; ++column) {
      sum += weights[column] * input[column];
    }
  }
#endif
  for (; column < size; ++column) {
    sum += weights[column] * input[column];
  }
  return sum;
}

// Pair products in int16 before widening. Inputs are in [-127, 127], so even
// weights of -128 give at most 2 * 128 * 127 = 32512, which fits int16.
int32_t dot_q8_block(const int8_t* weights, const int8_t* input) {
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
  const int8x16_t weights0 = vld1q_s8(weights);
  const int8x16_t weights1 = vld1q_s8(weights + 16);
  const int8x16_t input0 = vld1q_s8(input);
  const int8x16_t input1 = vld1q_s8(input + 16);
  int16x8_t products0 = vmull_s8(vget_low_s8(weights0), vget_low_s8(input0));
  products0 = vmlal_s8(products0, vget_high_s8(weights0), vget_high_s8(input0));
  int16x8_t products1 = vmull_s8(vget_low_s8(weights1), vget_low_s8(input1));
  products1 = vmlal_s8(products1, vget_high_s8(weights1), vget_high_s8(input1));
  int32x4_t sums = vpaddlq_s16(products0);
  sums = vpadalq_s16(sums, products1);
  const int32x2_t pair = vadd_s32(vget_low_s32(sums), vget_high_s32(sums));
  return vget_lane_s32(vpadd_s32(pair, pair), 0);
#else
  int32_t sum = 0;
  for (int i = 0; i < kQuantGroup; i++) {
    sum += weights[i] * input[i];
  }
  return sum;
#endif
}

// std::lround without the library call: truncate, then step away from zero at
// a remainder of one half. The remainder is exact for |value| < 2^23.
int32_t round_half_away(float value) {
  int32_t whole = static_cast<int32_t>(value);
  const float remainder = value - static_cast<float>(whole);
  if (remainder >= 0.5f) {
    ++whole;
  } else if (remainder <= -0.5f) {
    --whole;
  }
  return whole;
}

void quantize_block_scalar(const float* input, int count, Q8Block& block) {
  float largest = 0;
  for (int i = 0; i < count; i++) {
    largest = std::fmax(largest, std::fabs(input[i]));
  }
  block.scale = largest / 127.0f;
  // A zero group would divide by zero; its values are already exactly zero.
  const float inverse = largest > 0 ? 127.0f / largest : 0.0f;
  for (int i = 0; i < count; i++) {
    block.values[i] =
        static_cast<int8_t>(std::clamp(round_half_away(input[i] * inverse), -127, 127));
  }
  for (int i = count; i < kQuantGroup; i++) {
    block.values[i] = 0;
  }
}

#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
int32x4_t round_half_away(float32x4_t value) {
  int32x4_t whole = vcvtq_s32_f32(value);
  const float32x4_t remainder = vsubq_f32(value, vcvtq_f32_s32(whole));
  // Comparison masks are -1 where true.
  whole = vsubq_s32(whole, vreinterpretq_s32_u32(vcgeq_f32(remainder, vdupq_n_f32(0.5f))));
  whole = vaddq_s32(whole, vreinterpretq_s32_u32(vcleq_f32(remainder, vdupq_n_f32(-0.5f))));
  return vmaxq_s32(vminq_s32(whole, vdupq_n_s32(127)), vdupq_n_s32(-127));
}

// Same result as quantize_block_scalar for a full block.
void quantize_block_neon(const float* input, Q8Block& block) {
  float32x4_t values[8];
  float32x4_t largest = vdupq_n_f32(0);
  for (int i = 0; i < 8; ++i) {
    values[i] = vld1q_f32(input + 4 * i);
    largest = vmaxq_f32(largest, vabsq_f32(values[i]));
  }
  float32x2_t pair = vpmax_f32(vget_low_f32(largest), vget_high_f32(largest));
  pair = vpmax_f32(pair, pair);
  const float maximum = vget_lane_f32(pair, 0);
  block.scale = maximum / 127.0f;
  const float inverse = maximum > 0 ? 127.0f / maximum : 0.0f;
  int16x8_t narrowed[4];
  for (int i = 0; i < 4; ++i) {
    narrowed[i] = vcombine_s16(vmovn_s32(round_half_away(vmulq_n_f32(values[2 * i], inverse))),
                               vmovn_s32(round_half_away(vmulq_n_f32(values[2 * i + 1], inverse))));
  }
  vst1q_s8(block.values, vcombine_s8(vmovn_s16(narrowed[0]), vmovn_s16(narrowed[1])));
  vst1q_s8(block.values + 16, vcombine_s8(vmovn_s16(narrowed[2]), vmovn_s16(narrowed[3])));
}

// e^x = 2^n * e^r with |r| <= ln(2) / 2; Cephes polynomial for e^r. Inputs are
// clamped so that 2^n stays a normal float.
float32x4_t exp_neon(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.3f));
  const float32x4_t scaled = vaddq_f32(vmulq_n_f32(x, 1.44269504088896341f), vdupq_n_f32(0.5f));
  int32x4_t n = vcvtq_s32_f32(scaled);
  // Truncation rounds negative values up; step back to the floor.
  n = vaddq_s32(n, vreinterpretq_s32_u32(vcgtq_f32(vcvtq_f32_s32(n), scaled)));
  const float32x4_t whole = vcvtq_f32_s32(n);
  float32x4_t r = vsubq_f32(x, vmulq_n_f32(whole, 0.693359375f));
  r = vsubq_f32(r, vmulq_n_f32(whole, -2.12194440e-4f));
  float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
  y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(1.3981999507e-3f));
  y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(8.3334519073e-3f));
  y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(4.1665795894e-2f));
  y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(1.6666665459e-1f));
  y = vaddq_f32(vmulq_f32(y, r), vdupq_n_f32(5.0000001201e-1f));
  y = vaddq_f32(vaddq_f32(vmulq_f32(y, vmulq_f32(r, r)), r), vdupq_n_f32(1.0f));
  const int32x4_t exponent = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(exponent));
}
#endif

}  // namespace

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

void matvec(const float* input, Matrix weight, float* output) {
  const float* rows = reinterpret_cast<const float*>(weight.data);
  for (int row = 0; row < weight.rows; ++row) {
    const float* weights = rows + static_cast<size_t>(row) * weight.columns;
    output[row] = dot_fp32(weights, input, weight.columns);
  }
}

void quantize_q8(const float* input, int size, Q8Block* output) {
  const int blocks = q8_blocks(size);
  for (int b = 0; b < blocks; b++) {
    const int base = b * kQuantGroup;
    const int count = std::min(kQuantGroup, size - base);
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
    if (count == kQuantGroup) {
      quantize_block_neon(input + base, output[b]);
      continue;
    }
#endif
    quantize_block_scalar(input + base, count, output[b]);
  }
}

// Each group is an exact integer dot, scaled by both groups' scales.
void matvec_q8(const Q8Block* input, Matrix weight, float* output) {
  const int blocks = q8_blocks(weight.columns);
  const Q8Block* rows = reinterpret_cast<const Q8Block*>(weight.data);
  for (int row = 0; row < weight.rows; ++row) {
    const Q8Block* groups = rows + static_cast<size_t>(row) * blocks;
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
      __builtin_prefetch(reinterpret_cast<const char*>(groups + b) + 256);
      const int32_t dot = dot_q8_block(groups[b].values, input[b].values);
      sum += static_cast<float>(dot) * (groups[b].scale * input[b].scale);
    }
    output[row] = sum;
  }
}

// Keep a weight row hot while applying it to every token in the batch.
void matmul(const float* input, Matrix weight, int count, int output_stride, float* output) {
  const float* rows = reinterpret_cast<const float*>(weight.data);
  for (int row = 0; row < weight.rows; ++row) {
    const float* weights = rows + static_cast<size_t>(row) * weight.columns;
    for (int token = 0; token < count; ++token) {
      output[static_cast<size_t>(token) * output_stride + row] =
          dot_fp32(weights, input + static_cast<size_t>(token) * weight.columns, weight.columns);
    }
  }
}

void matmul_q8(const Q8Block* input, Matrix weight, int count, int output_stride, float* output) {
  const int blocks = q8_blocks(weight.columns);
  const Q8Block* rows = reinterpret_cast<const Q8Block*>(weight.data);
  for (int row = 0; row < weight.rows; ++row) {
    const Q8Block* weights = rows + static_cast<size_t>(row) * blocks;
    for (int token = 0; token < count; ++token) {
      const Q8Block* values = input + static_cast<size_t>(token) * blocks;
      float sum = 0;
      for (int b = 0; b < blocks; ++b) {
        __builtin_prefetch(reinterpret_cast<const char*>(weights + b) + 256);
        const int32_t dot = dot_q8_block(weights[b].values, values[b].values);
        sum += static_cast<float>(dot) * (weights[b].scale * values[b].scale);
      }
      output[static_cast<size_t>(token) * output_stride + row] = sum;
    }
  }
}

void embedding_lookup(Matrix table, int token, float* output) {
  const std::byte* row =
      table.data + static_cast<size_t>(token) * row_bytes(table.format, table.columns);
  if (table.format == Format::kQ8_0) {
    const Q8Block* blocks = reinterpret_cast<const Q8Block*>(row);
    for (int i = 0; i < table.columns; ++i) {
      const Q8Block& block = blocks[i / kQuantGroup];
      output[i] = block.scale * block.values[i % kQuantGroup];
    }
  } else {
    std::memcpy(output, row, table.columns * sizeof(float));
  }
}

void rope_angles(int position, int head_size, float theta, float* cosines, float* sines) {
  for (int pair = 0; pair < head_size / 2; pair++) {
    const float frequency = 1.0f / std::pow(theta, 2 * pair / static_cast<float>(head_size));
    const float angle = position * frequency;
    cosines[pair] = std::cos(angle);
    sines[pair] = std::sin(angle);
  }
}

void rope_inplace(const float* cosines, const float* sines, int head_size, int query_size,
                  int key_size, float* query, float* key) {
  for (int i = 0; i < query_size; i += 2) {
    const int pair = (i % head_size) / 2;
    const float cosine = cosines[pair];
    const float sine = sines[pair];
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
  int i = 0;
#if defined(__ARM_NEON) && !defined(UNREMARKABLE_SCALAR)
  for (; i + 4 <= size; i += 4) {
    const float32x4_t value = vld1q_f32(gate + i);
    const float32x4_t denominator = vaddq_f32(vdupq_n_f32(1.0f), exp_neon(vnegq_f32(value)));
    // Two Newton steps refine the 8-bit reciprocal estimate to full precision.
    float32x4_t reciprocal = vrecpeq_f32(denominator);
    reciprocal = vmulq_f32(vrecpsq_f32(denominator, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(denominator, reciprocal), reciprocal);
    vst1q_f32(gate + i, vmulq_f32(vmulq_f32(value, reciprocal), vld1q_f32(up + i)));
  }
#endif
  for (; i < size; i++) {
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
