#ifndef UNREMARKABLE_SRC_KERNELS_H_
#define UNREMARKABLE_SRC_KERNELS_H_

#include <cstdint>

namespace unremarkable {

// Q8_0: thirty-two values share one FP32 scale, stored ahead of them so a row
// reads as one sequential stream.
inline constexpr int kQuantGroup = 32;

struct Q8Block {
  float scale;
  int8_t values[kQuantGroup];
};

// Blocks in one row of `columns` values. The last block zero-pads, and zeros
// contribute nothing to a dot product, so tails need no special case.
constexpr int q8_blocks(int columns) { return (columns + kQuantGroup - 1) / kQuantGroup; }

// Quantize [size] values, symmetric about zero and scaled by the group maximum.
void quantize_q8(const float* input, int size, Q8Block* output);

// Expand one quantized row into [size] floats.
void dequantize_q8(const Q8Block* input, int size, float* output);

// As matvec with both operands Q8_0. Input values must be in [-127, 127], as
// produced by quantize_q8. The integer dot within each group is exact; groups
// then accumulate in FP32, so every target agrees.
void matvec_q8(const Q8Block* input, const Q8Block* weight, int columns, int rows, float* output);

// Build without fast-math or FMA contraction. ARM matvec uses four NEON vector
// accumulators; define UNREMARKABLE_SCALAR for index-order scalar accumulation.
// rmsnorm allows output == input; matvec does not.
void rmsnorm(const float* input, const float* weight, int size, float* output);
void softmax_inplace(float* values, int size);
void matvec(const float* input, const float* weight, int columns, int rows, float* output);

// Copy one row of the embedding table [vocabulary][dim] into output [dim].
void embedding_lookup(const float* table, int token, int dim, float* output);

// Rotate query [query_size] and key [key_size] in place, pairing adjacent
// components. Head size must be even; both vector sizes must be multiples of it,
// with key_size <= query_size. theta is the RoPE base frequency.
void rope_inplace(int position, int head_size, int query_size, int key_size, float theta,
                  float* query, float* key);

// Attend to cache positions 0..position, including the current token. Query and
// output are [n_heads][head_size]; caches are [context][n_kv_heads][head_size].
// scores is reusable scratch [n_heads][context]. n_heads must be a multiple of
// n_kv_heads, and position must fit context. Buffers must not overlap.
void causal_attention(const float* query, const float* key_cache, const float* value_cache,
                      int n_heads, int n_kv_heads, int head_size, int context, int position,
                      float* scores, float* output);

// gate[i] = silu(gate[i]) * up[i]. The two buffers are [size].
void swiglu_inplace(const float* up, int size, float* gate);

// output[i] += input[i]. Both buffers are [size]; exact aliasing is allowed.
void add_inplace(const float* input, int size, float* output);

}  // namespace unremarkable
#endif
