# One token through the naive engine

Read this alongside [engine.cpp](../src/engine.cpp). The example dimensions are
SmolLM2-135M's: 30 layers, activation width 576, feed-forward width 1536,
9 query heads, 3 key/value heads, head width 64, vocabulary size 49152.
TinyStories follows the same path with different dimensions.

## Three kinds of memory

| Owner | Contents | What survives? |
| --- | --- | --- |
| `Model` | The weight payload, FP32 or Q8_0, and read-only matrix views | The whole engine lifetime |
| `KVCache` | Each layer's keys and values, `[layer, context, kv_dim]` | Earlier positions in the current sequence |
| `Scratch` | Residual, normalized input, query, attention output, projected result, gate, up, scores, logits, quantized input | Storage survives; contents are overwritten |

All three have backing storage in process RAM. Cache lines move through the CPU
cache hierarchy as instructions read and write them. The code owns buffers and
controls access order; it does not pin a vector to L1 or a matrix to L2.

For SmolLM2 at context 512, weights occupy about 513 MiB as FP32 or 144 MiB as
Q8_0, and K/V storage occupies `30 × 512 × 192 × 2 × 4 = 23,592,960` bytes, or
22.5 MiB. A residual vector is only `576 × 4 = 2304` bytes. Named buffers keep
their roles visible; `projected` is reused for both residual additions. This
version uses one extra dim-sized buffer versus the original engine to separate
normalization from attention output.

## 1. Load once per engine process

[model.cpp](../src/model.cpp) reads the header, validates the expected payload
size, allocates one weight buffer, and reads the payload into it. It then binds
`LayerWeights` and `Matrix` views to portions of that buffer. Views copy pointers
and dimensions, never the matrices themselves.

The on-disk layout groups weights by operation: all layers' Q matrices are stored
together, then all layers' K matrices, and so on. The loader translates that into
`model.layers[layer].query`, `.key`, `.value`, etc., once at startup. The forward
pass therefore needs no checkpoint offset arithmetic.

Supported files are legacy llama2.c, tagged UNRK version 1 (FP32), and version 2
(FP32 or Q8_0) checkpoints. All dimensions and byte counts are checked before
allocating the weight payload. There is no whole-model read from storage inside
`forward()`.

## 2. Turn text into tokens

[main.cpp](../src/main.cpp) wraps chat prompts when required, then calls the
[tokenizer](../src/tokenizer.cpp). Tokens are integer IDs, not activation vectors.
The CLI prefills by calling `forward(token, position)` for each prompt token in
order. This baseline uses the same one-token path for prefill and decode.

## 3. Look up the embedding

`embedding_lookup()` copies the selected embedding row into `scratch.residual`,
expanding it from Q8_0 when needed. For this model that gives `residual[576]`: the
current token's running activation. Its contents change as it passes through the
layers.

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

## Zoom in: one matrix-vector multiplication

The `Matrix` argument carries a read-only pointer, row count, and column count.
Without the prefetch hint added in rung 02, the entire FP32 implementation is:

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
weights. For Q, the input is 2.25 KiB and the matrix is about 1.27 MiB.
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

Waiting for those misses dominates. At 1.08 tokens/s the loop pulls weights at
about 40% of the rate one core can stream. Rung 02 therefore walks each row one
64-byte line (16 floats) at a time and issues `__builtin_prefetch` for the line
256 bytes ahead, so the next miss is already in flight while the current line is
summed. The additions keep their order, so results are bit-identical, and decode
rises to 1.44 tokens/s ([measurements](optimizations.md#measured-02-on-the-tablet-2026-09-12)).

Rung 03 then speeds up the arithmetic. With most waits hidden, the single running
`sum` becomes the limit, because each addition must wait for the one before it.
NEON instructions multiply four adjacent values at once into four accumulators,
16 partial sums in all, so the additions overlap and decode reaches 1.75 tokens/s
([measurements](optimizations.md#measured-03-on-the-tablet-2026-09-12)). Adding
in a different order makes the logits differ slightly from the scalar loop (at
most 2e-4 on SmolLM2); building with `-DUNREMARKABLE_SCALAR` restores rung 02's
loop.

Rung 04 reads fewer bytes. A Q8_0 matrix stores each row as blocks of 32 int8
weights with one FP32 scale, and `Engine::project()` first quantizes the input
into the same blocks. Each block is then an exact integer dot product, scaled by
both blocks' scales, and the blocks add in FP32. SmolLM2's matrices shrink from
538 MB to 151 MB, and decode reaches 3.80 tokens/s
([measurements](optimizations.md#measured-04-on-the-tablet-2026-09-12)).
Quantization moves the logits (0.19 on average) and changes the generated text.

The code has no threads. Normal compiler optimization is enabled; the source is
not a guarantee about every machine instruction a compiler may generate.

## 5. Finish the token, then repeat

After all layers, normalize the residual in place and multiply by the classifier
matrix `[49152, 576]`. The result is `logits[49152]`: one score per vocabulary ID.
`forward()` returns a span borrowing that output buffer.

The [sampler](../src/sampler.cpp) selects an ID (argmax for greedy decoding).
The tokenizer decodes it to text, which the CLI prints. Unless generation stops,
that chosen ID becomes the next call's input at the next position.

The KV cache avoids running previous tokens through the network again, but the
new token still uses all the projection weights. SmolLM2 traverses about 538 MB
of FP32 matrix weights per token, or 151 MB as Q8_0; most cannot survive in CPU
caches between passes. KV reads grow with the length of the sequence.

`reset()` restarts position counting. We keep allocations and weights, overwrite
each new K/V slot before use, and never read the old sequence's later slots.
The supplied CLI starts a new engine process for each invocation.
