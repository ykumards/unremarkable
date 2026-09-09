#!/usr/bin/env python3
"""Convert a Hugging Face Llama-architecture model to the unremarkable format.

Writes MODEL.bin (weights) and MODEL.tok (byte-level BPE tokenizer). Reads
safetensors and tokenizer.json directly; no torch or numpy required.
"""
import argparse
import json
from pathlib import Path
import struct

MODEL_MAGIC = 0x4B524E55  # "UNRK"
TOKEN_MAGIC = 0x4B4F5455  # "UTOK"
VERSION = 1


def bf16_to_f32(raw):
    """BF16 is the high 16 bits of FP32, so widening is a byte interleave."""
    out = bytearray(len(raw) * 2)
    out[2::4] = raw[0::2]
    out[3::4] = raw[1::2]
    return bytes(out)


class Safetensors:
    def __init__(self, path):
        self.stream = path.open("rb")
        length = struct.unpack("<Q", self.stream.read(8))[0]
        self.header = json.loads(self.stream.read(length))
        self.header.pop("__metadata__", None)
        self.base = 8 + length

    def tensor(self, name, shape):
        entry = self.header[name]
        if entry["shape"] != list(shape):
            raise ValueError(f"{name}: expected shape {list(shape)}, found {entry['shape']}")
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, found {entry['dtype']}")
        start, end = entry["data_offsets"]
        self.stream.seek(self.base + start)
        return bf16_to_f32(self.stream.read(end - start))


def interleave_rope(raw, rows, columns, head_size):
    """HF rotates halves of a head; this engine rotates adjacent pairs.

    Reorder each head's rows from [first half, second half] to interleaved, so
    rope_inplace reproduces the reference rotation. Mirrors the inverse of the
    permutation the HF Llama conversion applies to q_proj and k_proj.
    """
    stride = columns * 4
    half = head_size // 2
    out = bytearray(len(raw))
    for head in range(rows // head_size):
        top = head * head_size
        for i in range(half):
            source = (top + i) * stride
            out[(top + 2 * i) * stride:(top + 2 * i + 1) * stride] = raw[source:source + stride]
            source = (top + half + i) * stride
            out[(top + 2 * i + 1) * stride:(top + 2 * i + 2) * stride] = raw[source:source + stride]
    return bytes(out)


def export_model(source, output, context):
    config = json.loads((source / "config.json").read_text())
    if config["architectures"] != ["LlamaForCausalLM"]:
        raise ValueError(f"unsupported architecture: {config['architectures']}")
    if config["hidden_act"] != "silu":
        raise ValueError(f"unsupported activation: {config['hidden_act']}")
    dim = config["hidden_size"]
    hidden = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    vocab = config["vocab_size"]
    shared = config["tie_word_embeddings"]
    head_size = dim // heads
    kv_dim = head_size * kv_heads
    context = min(context, config["max_position_embeddings"])
    if dim % heads or heads % kv_heads or head_size % 2:
        raise ValueError("model dimensions are incompatible with this engine")

    weights = Safetensors(source / "model.safetensors")
    layer = lambda i, name: f"model.layers.{i}.{name}.weight"

    def each(name, rows, permute=False):
        for i in range(layers):
            raw = weights.tensor(layer(i, name), (rows, dim))
            yield interleave_rope(raw, rows, dim, head_size) if permute else raw

    with output.open("wb") as out:
        out.write(struct.pack("<Ii7if", MODEL_MAGIC, VERSION, dim, hidden, layers, heads,
                              kv_heads, vocab if shared else -vocab, context,
                              float(config["rope_theta"])))
        out.write(weights.tensor("model.embed_tokens.weight", (vocab, dim)))
        for i in range(layers):
            out.write(weights.tensor(layer(i, "input_layernorm"), (dim,)))
        for raw in each("self_attn.q_proj", dim, permute=True):
            out.write(raw)
        for raw in each("self_attn.k_proj", kv_dim, permute=True):
            out.write(raw)
        for raw in each("self_attn.v_proj", kv_dim):
            out.write(raw)
        for i in range(layers):
            out.write(weights.tensor(layer(i, "self_attn.o_proj"), (dim, dim)))
        for i in range(layers):
            out.write(weights.tensor(layer(i, "post_attention_layernorm"), (dim,)))
        for raw in each("mlp.gate_proj", hidden):
            out.write(raw)
        for i in range(layers):
            out.write(weights.tensor(layer(i, "mlp.down_proj"), (dim, hidden)))
        for raw in each("mlp.up_proj", hidden):
            out.write(raw)
        out.write(weights.tensor("model.norm.weight", (dim,)))
        if not shared:
            out.write(weights.tensor("lm_head.weight", (vocab, dim)))
    return dim, layers, vocab, context


def byte_to_unicode():
    """GPT-2's reversible byte/codepoint map, as used by the ByteLevel model."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped, extra = list(printable), 0
    for value in range(256):
        if value not in printable:
            printable.append(value)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, (chr(code) for code in mapped)))


def export_tokenizer(source, output, vocab_size):
    spec = json.loads((source / "tokenizer.json").read_text())
    model = spec["model"]
    if model["type"] != "BPE" or model.get("byte_fallback"):
        raise ValueError("expected a byte-level BPE tokenizer")
    if spec.get("normalizer") or spec.get("post_processor"):
        raise ValueError("normalizers and post-processors are not supported")
    vocab = model["vocab"]
    if len(vocab) != vocab_size:
        raise ValueError(f"tokenizer has {len(vocab)} entries, model expects {vocab_size}")
    special = {token["content"] for token in spec["added_tokens"] if token["special"]}
    decode = {code: value for value, code in byte_to_unicode().items()}

    # Byte-level BPE merges by lowest rank; the engine merges by highest score,
    # so rank is negated. Each merge produces exactly one vocabulary entry.
    score = {}
    for rank, merge in enumerate(model["merges"]):
        pair = merge.split(" ") if isinstance(merge, str) else merge
        joined = "".join(pair)
        if joined not in vocab:
            raise ValueError(f"merge result missing from vocabulary: {joined!r}")
        if vocab[joined] in score:
            raise ValueError(f"vocabulary entry produced by two merges: {joined!r}")
        score[vocab[joined]] = -float(rank)

    entries = [None] * vocab_size
    for text, identifier in vocab.items():
        if text in special:
            raw, flags = text.encode(), 1
        else:
            if any(code not in decode for code in text):
                raise ValueError(f"vocabulary entry outside the byte-level map: {text!r}")
            raw, flags = bytes(decode[code] for code in text), 0
        entries[identifier] = (score.get(identifier, -1e30), raw, flags)
    if any(entry is None for entry in entries):
        raise ValueError("vocabulary identifiers are not contiguous")

    longest = max(len(raw) for _, raw, _ in entries)
    with output.open("wb") as out:
        out.write(struct.pack("<IiiI", TOKEN_MAGIC, VERSION, vocab_size, longest))
        for value, raw, flags in entries:
            out.write(struct.pack("<fiB", value, len(raw), flags))
            out.write(raw)
    return longest, sum(1 for _, _, flags in entries if flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory holding config.json, "
                                                  "model.safetensors, and tokenizer.json")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output checkpoint path; the tokenizer replaces .bin with .tok")
    parser.add_argument("-c", "--context", type=int, default=2048,
                        help="context to store in the header (default 2048)")
    args = parser.parse_args()
    try:
        dim, layers, vocab, context = export_model(args.source, args.output, args.context)
        tokenizer = args.output.with_suffix(".tok")
        longest, specials = export_tokenizer(args.source, tokenizer, vocab)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    size = args.output.stat().st_size / 1048576
    print(f"Wrote {args.output} ({size:.1f} MiB): dim {dim}, {layers} layers, "
          f"vocab {vocab}, context {context}")
    print(f"Wrote {tokenizer}: {vocab} tokens, {specials} special, longest {longest} bytes")


if __name__ == "__main__":
    main()
