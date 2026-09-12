#ifndef UNREMARKABLE_SRC_KERNELS_H_
#define UNREMARKABLE_SRC_KERNELS_H_

namespace unremarkable {

struct Matrix;
struct Q8Block;

// Build without fast-math or FMA contraction. rmsnorm allows output == input;
// matvec does not.
void rmsnorm(const float* input, const float* weight, int size, float* output);
void softmax_inplace(float* values, int size);
// input: [weight.columns], reused for every row. weight: row-major FP32.
// output: [weight.rows], overwritten. No allocations; buffers must not overlap.
void matvec(const float* input, Matrix weight, float* output);

// Quantize [size] values into q8_blocks(size) blocks, symmetric about zero.
void quantize_q8(const float* input, int size, Q8Block* output);
// As matvec for a Q8_0 weight, with the input from quantize_q8. The integer dot
// within each group is exact; groups accumulate in FP32.
void matvec_q8(const Q8Block* input, Matrix weight, float* output);

// Copy row `token` of the embedding table into output [table.columns].
void embedding_lookup(Matrix table, int token, float* output);

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
