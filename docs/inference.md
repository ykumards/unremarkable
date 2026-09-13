# Inference

[Engine::forward](../src/engine.cpp) takes a token ID and position, updates the
KV cache, and returns next-token logits. These examples use SmolLM2-135M:
30 layers, activation width 576, feed-forward width 1536, 9 query heads,
3 key/value heads, head width 64, vocabulary size 49152.
TinyStories follows the same path with different dimensions.

## Memory

| Owner | Contents | What survives? |
| --- | --- | --- |
| `Model` | FP32 or Q8_0 weights and read-only matrix views | The whole engine lifetime |
| `KVCache` | Each layer's keys and values, `[layer, context, kv_dim]` | Earlier positions in the current sequence |
| `Scratch` | Residual, normalized input, query, attention output, projected result, gate, up, scores, logits, quantized input | Storage survives; contents are overwritten |

All buffers live in process RAM. The CPU caches their contents as needed; the
code controls access order, not which cache holds each buffer.

For SmolLM2 at context 512, weights occupy about 513 MiB as FP32 or 144 MiB as
Q8_0. K/V storage occupies `30 × 512 × 192 × 2 × 4 = 23,592,960` bytes, or 22.5 MiB. A residual vector is
only `576 × 4 = 2304` bytes. `projected` is reused for both residual additions.

## 1. Load weights

[model.cpp](../src/model.cpp) reads the header, validates the expected payload
size, allocates one weight buffer, and reads the payload into it. It then binds
`LayerWeights` and `Matrix` views to portions of that buffer. Each view holds
a pointer and dimensions; the weights stay in the original buffer.

The on-disk layout groups weights by operation: all layers' Q matrices are stored
together, then all layers' K matrices, and so on. The loader translates that into
`model.layers[layer].query`, `.key`, `.value`, etc., once at startup.

Supported files are legacy llama2.c, UNRK version 1 (FP32), and UNRK version 2
(FP32 or Q8_0) checkpoints.
Weights are loaded once per engine process, not on each forward pass.

## 2. Tokenize the prompt

[main.cpp](../src/main.cpp) wraps chat prompts when required, then calls the
[tokenizer](../src/tokenizer.cpp). Each token is an integer ID.
The CLI calls `Engine::prefill()` with the prompt IDs. It processes chunks of
up to eight tokens, then returns the last token's logits. Decode calls
`forward(token, position)` once per generated token, as described below.

### Batched prefill

[prefill.cpp](../src/prefill.cpp) follows the same layer operations, with a row
of activations per prompt token. For a chunk of eight:

```text
Decode:  x[576]     × W[rows, 576]ᵀ → y[rows]       (GEMV)
Prefill: X[8, 576]  × W[rows, 576]ᵀ → Y[8, rows]    (GEMM)
```

We know all prompt IDs in advance. Each chunk passes through all layers before
starting the next chunk. Within a layer, Q/K/V and feed-forward projections
process the whole chunk. RoPE uses each token's absolute position; attention
only reads K/V through that position, even if later slots are already written.
The cache layout is unchanged.

The first GEMM kernel loops over weight rows, then tokens, then columns (Q8
blocks for quantized weights). Each row is reused from cache across the chunk;
the dot product still loads it into registers for each token. There is no packed
weight format or register tiling yet. Both cores split output columns of `Y`,
so one worker wake/wait covers a projection for the whole chunk.

Only the final prompt token needs vocabulary scores. Earlier tokens skip the
final norm and classifier; their layer K/V entries are still computed. With
Q8 weights, the eight-slot scratch adds about 175 KiB for SmolLM2. It is allocated
once, and decode reuses the first slot. Attention scores and logits stay unbatched.

`-b 8` is the default. `-b 1` isolates skipping unused vocabulary scores;
`-b 0` runs the original `forward()` loop, including every classifier. Neither
option changes decode. The returned logits are borrowed until the next
`forward()` or `prefill()` call.

## 3. Look up the embedding

`embedding_lookup()` copies the selected embedding row into `scratch.residual`,
expanding Q8_0 weights to FP32 when needed.
For this model that gives `residual[576]`: the current token's running activation.
Its contents change as it passes through the layers.

## 4. Execute one layer

The loop in `Engine::forward()` performs these operations in order:

| Operation | Reads | Writes |
| --- | --- | --- |
| Attention RMSNorm | Residual + norm weights | `normalized[576]` |
| Q projection | Normalized + Q matrix | `query[576]` |
| K projection | Normalized + K matrix | Current `key[192]` in cache |
| V projection | Normalized + V matrix | Current `value[192]` in cache |
| RoPE | Current Q and K, position | Q and current K in place |
| Attention | Q + K/V history through this position | `attention_output[576]` and scores |
| Output projection | Attention output + output matrix | `projected[576]` |
| Residual addition | Projected + residual | Residual in place |
| Feed-forward RMSNorm | Residual + norm weights | Normalized |
| Gate and up projections | Normalized + two matrices | `gate[1536]`, `up[1536]` |
| SwiGLU | Gate and up | Gate in place |
| Down projection | Gate + down matrix | Projected |
| Residual addition | Projected + residual | Residual in place |

K/V go directly into their final cache slots. Each layer has its own history;
attention never reads beyond the current position. Query heads share K/V heads
in groups of three. The keys are cached after RoPE, while values are unchanged.

Within each head, query–key dots and softmax remain scalar. The square root of
head width is computed once per kernel call. NEON multiplies and adds four
value components at a time; each component still accumulates positions in order.

## Matrix-vector multiplication

The `Matrix` argument carries a read-only pointer, dimensions, and weight format.
Rung 01 computes each row with one running sum:

```cpp
for (int row = 0; row < weight.rows; ++row) {
    const float* weights = weight.data + size_t(row) * weight.columns;
    float sum = 0;
    for (int column = 0; column < weight.columns; ++column) {
        sum += weights[column] * input[column];
    }
    output[row] = sum;
}
```

Every row reuses the same input vector but reads a new contiguous range of
weights. For FP32 Q, the input is 2.25 KiB and the matrix is about 1.27 MiB.
On the tablet, the input is small enough for its 32 KiB per-core L1 data cache;
the matrix exceeds the shared 512 KiB L2. Residency depends on other accesses.

```text
RAM-backed weight buffer -> CPU caches -> weight in a register
RAM-backed input buffer  -> CPU caches -> input in a register
                                           |
                                      multiply + add
                                           |
                                      sum in a register
                                           |
                               output store through caches
```

A cache miss brings in a line containing multiple adjacent values. As rows
advance, older weight lines can be evicted. The input is repeatedly reused and
likely to remain cached. Stores update the output through caches; they need not
be written all the way back to DRAM before the next operation can use them.

Rung 02 adds `__builtin_prefetch`: request weights 256 bytes ahead, once per
16 floats, while summing the current block. Addition order stays unchanged.

Rung 03 uses four NEON vectors to hold 16 partial sums. Each multiply handles
four values; separate sums shorten the dependency chain. Combining them changes
floating-point addition order, so logits can differ slightly.
`-DUNREMARKABLE_SCALAR` selects rung 02's FP32 loop.

Rung 04 stores weights in Q8_0 blocks: 32 int8 values and one FP32 scale.
`Engine::project()` quantizes the input into the same format, computes integer
dot products per block, then multiplies by both scales and adds in FP32.
Norm vectors, the KV cache, and projection outputs stay FP32.

Rung 06 changes the integer dot product inside each Q8 block. It uses two
`vmull_s8` and two `vmlal_s8` instructions to form pairs of products in int16,
then one `vpaddlq_s16` and one `vpadalq_s16` to sum into int32. That is six
arithmetic instructions instead of eight, excluding loads, the final lane sum,
and scale application. The FP32 accumulation order stays unchanged.

The largest pair has magnitude `2 × 128 × 127 = 32512`, within int16. This relies
on activations being in `[-127, 127]`, as produced by `quantize_q8`; weights may
include `-128`.

Rungs 01–04 use one thread. Rung 05 adds `-j 2`:

1. `Engine::project()` quantizes the input once for Q8.
2. The caller computes rows `[0, rows/2)`; a persistent worker computes the rest.
3. The caller waits for the worker before returning or reusing the input buffer.

Both threads read the same weights and input. Each writes separate output rows;
no matrix or activation copy is needed. Row boundaries include the full Q8 blocks
and scales. Each dot product keeps its addition order, so logits match `-j 1`.
Projections with fewer than 64 rows stay on the caller. Quantization, norms,
RoPE, and sampling also stay on the caller.

Rung 09 also splits attention heads through `Engine::attend()`: four heads on
the caller and five on the worker for SmolLM2. They read the same K/V cache and
write disjoint score and output slices. No reduction between threads is needed.
The caller waits before the output projection; batched prefill does this for
each query position in turn. With `-j 1`, all heads stay on the caller.

[worker.cpp](../src/worker.cpp) contains the wake/wait logic. The worker is created
once per engine and joined before its buffers are destroyed. `-j 1` (the default)
creates no worker. See the [measurements](optimizations.md) for speed and output
comparisons, including [Q8 accuracy](optimizations.md#accuracy-2026-09-13).

## 5. Sample the next token

After all layers, normalize the residual in place and multiply by the classifier
matrix `[49152, 576]`. The result is `logits[49152]`: one score per vocabulary ID.
`forward()` returns a span over this buffer; the next call overwrites it.

The [sampler](../src/sampler.cpp) selects an ID (argmax for greedy decoding).
The tokenizer decodes it to text, which the CLI prints. Unless generation stops,
that chosen ID becomes the next call's input at the next position.

The KV cache saves earlier tokens' keys and values. Each new token still reads
the projection weights: about 538 MB per token for SmolLM2 in FP32, or 151 MB
in Q8_0. Both exceed the CPU cache capacity. KV reads grow with the length of the sequence.

`reset()` restarts position counting. We keep allocations and weights, overwrite
each new K/V slot before use, and never read the old sequence's later slots.
The supplied CLI starts a new engine process for each invocation.
