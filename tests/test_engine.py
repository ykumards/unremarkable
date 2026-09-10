#!/usr/bin/env python3
"""End-to-end correctness checks. Requires `make test-models` once."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/stories260K.bin"
TOKENIZER = ROOT / "models/tok512.bin"
CHAT_TOKENIZER = ROOT / "models/smollm2-135m.tok"
MODEL_MAGIC = 0x4B524E55
QUANT_GROUP = 32  # Values per scale; must match kQuantGroup in src/kernels.h.

# Identifiers from the reference tokenizer for SmolLM2-135M-Instruct, including
# the text runs that tools/export_hf.py wraps in ChatML markers.
BYTE_LEVEL_TOKENS = [
    ("Hello world", [19556, 905]),
    ("What is the capital of France?", [1780, 314, 260, 3575, 282, 4649, 47]),
    ("The year 2024 had 365 days; 12 months.",
     [504, 713, 216, 34, 32, 34, 36, 761, 216, 35, 38, 37, 2009, 43, 216, 33, 34, 2704, 30]),
    ("  leading and   multiple   spaces  ", [216, 2899, 284, 256, 2701, 256, 5600, 256]),
    ("def f(n):\n    return n+1\n", [1604, 275, 24, 94, 727, 472, 1003, 304, 27, 33, 198]),
    ("Gr\u00fc\u00dfe, caf\u00e9!", [27302, 7170, 34878, 85, 28, 37366, 17]),
    ("Tab\there\nNewline\n\nDouble", [30064, 197, 1531, 198, 5529, 1311, 198, 198, 33743]),
    ("'s 't 're 've 'm 'll 'd", [506, 637, 100, 637, 257, 637, 307, 637, 93, 637, 764, 637, 84]),
    ("a,b,c...   ???!!!  <tag>",
     [81, 28, 82, 28, 83, 2026, 256, 9148, 16693, 29646, 216, 2067, 9204, 46]),
    ("Hello, \u4e16\u754c!", [19556, 28, 216, 7906, 240, 178, 239, 230, 17]),
    ("system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face",
     [9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411,
      407, 19712, 8182]),
    ("user\nWhat is the capital of France?", [4093, 198, 1780, 314, 260, 3575, 282, 4649, 47]),
    ("assistant\n", [520, 9531, 198]),
    ("\n", [198]),
]

# Published llama2.c test_all.py fixture, pinned revision in THIRD_PARTY.md.
EXPECTED = '''Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she
'''


def constant_model(path, chosen_token):
    """A tiny GQA model with an unshared classifier selecting one fixed token."""
    dim, hidden, layers, heads, kv_heads, vocab, context = 4, 8, 1, 2, 1, 512, 8
    kv = 2
    embedding = [1.0, 0.0, 0.0, 0.0] * vocab
    weights = (embedding + [1.0] * dim + [0.0] * (dim * dim + 2 * dim * kv + dim * dim)
               + [1.0] * dim + [0.0] * (3 * dim * hidden) + [1.0] * dim
               + [0.0] * (context * (dim // heads)))
    classifier = [0.0] * (vocab * dim)
    classifier[chosen_token * dim] = 1.0
    weights += classifier
    path.write_bytes(struct.pack("<7i", dim, hidden, layers, heads, kv_heads, -vocab, context)
                     + struct.pack(f"<{len(weights)}f", *weights))



def float32(value):
    """Round a Python float to the FP32 value the engine would hold."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantize_q8(values, columns):
    """Independent Q8_0, written against the format rather than the engine's code.

    Returns the encoded rows and the values the engine will compute with, so the
    oracle sees what the checkpoint holds, not the weights it came from.
    """
    encoded, seen = bytearray(), []
    for start in range(0, len(values), columns):
        row = values[start:start + columns]
        for base in range(0, columns, QUANT_GROUP):
            group = row[base:base + QUANT_GROUP]
            group = group + [0.0] * (QUANT_GROUP - len(group))
            largest = max(abs(value) for value in group)
            scale = float32(largest / 127.0)
            inverse = float32(127.0 / largest) if largest > 0 else 0.0
            quantized = []
            for value in group:
                scaled = float32(value * inverse)
                rounded = int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))
                quantized.append(max(-127, min(127, rounded)))
            encoded += struct.pack(f"<f{QUANT_GROUP}b", scale, *quantized)
            seen.extend(float32(scale * q) for q in quantized[:columns - base])
    return bytes(encoded), seen


def reference_model(path, shared, kv_heads, theta=10000.0, tagged=False, quantize=False,
                    dim=16, hidden=24):
    """Independent float64 forward oracle; tiny random checkpoint on disk.

    With quantize set the matrices are written as Q8_0, and the oracle quantizes
    each projection input as the engine does. Norm vectors stay FP32 either way.
    """
    layers, heads, vocab, context = 2, 4, 512, 6
    head = dim // heads
    kv = head * kv_heads
    rng = random.Random(17)
    blob = []

    def tensor(count, norm=False, columns=None):
        values = [rng.uniform(0.8, 1.2) if norm else rng.uniform(-0.3, 0.3)
                  for _ in range(count)]
        # The oracle reads precisely the float32 values serialized for C++.
        values = list(struct.unpack(f"<{count}f", struct.pack(f"<{count}f", *values)))
        if quantize and columns is not None:
            encoded, values = quantize_q8(values, columns)
            blob.append(encoded)
        else:
            blob.append(struct.pack(f"<{count}f", *values))
        return values

    embedding = tensor(vocab * dim, columns=dim)
    att_norm = tensor(layers * dim, norm=True)
    wq = tensor(layers * dim * dim, columns=dim)
    wk = tensor(layers * kv * dim, columns=dim)
    wv = tensor(layers * kv * dim, columns=dim)
    wo = tensor(layers * dim * dim, columns=dim)
    ffn_norm = tensor(layers * dim, norm=True)
    w1 = tensor(layers * hidden * dim, columns=dim)
    w2 = tensor(layers * dim * hidden, columns=hidden)
    w3 = tensor(layers * hidden * dim, columns=dim)
    final_norm = tensor(dim, norm=True)
    if not tagged:
        tensor(context * head)  # Legacy RoPE tables, unused.
    classifier = embedding if shared else tensor(vocab * dim, columns=dim)
    signed_vocab = vocab if shared else -vocab
    if quantize:
        header = struct.pack("<Ii7ifi", MODEL_MAGIC, 2, dim, hidden, layers, heads, kv_heads,
                             signed_vocab, context, theta, 1)
    elif tagged:
        header = struct.pack("<Ii7if", MODEL_MAGIC, 1, dim, hidden, layers, heads, kv_heads,
                             signed_vocab, context, theta)
    else:
        header = struct.pack("<7i", dim, hidden, layers, heads, kv_heads, signed_vocab, context)
    path.write_bytes(header + b"".join(blob))
    keys, values = [[] for _ in range(layers)], [[] for _ in range(layers)]

    def norm(x, weights):
        scale = 1 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
        return [w * scale * v for v, w in zip(x, weights)]

    def mv(x, weights, rows, layer=0):
        columns = len(x)
        if quantize:
            _, x = quantize_q8(x, columns)  # The engine quantizes every projection input.
        start = layer * rows * columns
        return [sum(x[j] * weights[start + row * columns + j] for j in range(columns))
                for row in range(rows)]

    def rotate(x, pos):
        out = x.copy()
        for i in range(0, len(x), 2):
            angle = pos / (theta ** ((i % head) / head))
            c, s = math.cos(angle), math.sin(angle)
            out[i], out[i + 1] = x[i] * c - x[i + 1] * s, x[i] * s + x[i + 1] * c
        return out

    def forward(token, pos):
        x = embedding[token * dim:(token + 1) * dim]
        for layer in range(layers):
            normalized = norm(x, att_norm[layer * dim:(layer + 1) * dim])
            query = rotate(mv(normalized, wq, dim, layer), pos)
            keys[layer].append(rotate(mv(normalized, wk, kv, layer), pos))
            values[layer].append(mv(normalized, wv, kv, layer))
            attention = []
            for h in range(heads):
                offset = (h // (heads // kv_heads)) * head
                scores = [sum(query[h * head + j] * k[offset + j] for j in range(head))
                          / math.sqrt(head) for k in keys[layer]]
                probabilities = [math.exp(score - max(scores)) for score in scores]
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]
                attention.extend(sum(p * v[offset + j] for p, v in zip(probabilities, values[layer]))
                                 for j in range(head))
            x = [a + b for a, b in zip(x, mv(attention, wo, dim, layer))]
            normalized = norm(x, ffn_norm[layer * dim:(layer + 1) * dim])
            gate, up = mv(normalized, w1, hidden, layer), mv(normalized, w3, hidden, layer)
            activated = [g / (1 + math.exp(-g)) * u for g, u in zip(gate, up)]
            x = [a + b for a, b in zip(x, mv(activated, w2, dim, layer))]
        return mv(norm(x, final_norm), classifier, vocab)

    return forward


class EngineTests(unittest.TestCase):
    def test_forward_logits_against_independent_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "random.bin"
            for shared in (False, True):
                for kv_heads in (1, 2, 4):
                    with self.subTest(shared=shared, kv_heads=kv_heads):
                        reference = reference_model(model, shared, kv_heads)
                        tokens = [1, 19, 43, 7]
                        proc = subprocess.run([str(PROBE), str(model), *map(str, tokens)],
                                              capture_output=True, timeout=60)
                        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                        rows = proc.stdout.decode().splitlines()
                        self.assertEqual(len(rows), len(tokens))
                        for pos, row in enumerate(rows):
                            actual = list(map(float, row.split()))
                            expected = reference(tokens[pos], pos)
                            self.assertEqual(len(actual), len(expected))
                            for a, b in zip(actual, expected):
                                self.assertAlmostEqual(a, b, delta=2e-5)


    def test_quantized_checkpoint_matches_reference(self):
        """Q8_0 rows carry inline scales, so every weight stride changes with them."""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "quantized.bin"
            # dim 64 gives whole blocks; hidden 72 leaves a padded tail in w2.
            for shared in (False, True):
                for kv_heads in (1, 2, 4):
                    with self.subTest(shared=shared, kv_heads=kv_heads):
                        reference = reference_model(model, shared, kv_heads, tagged=True,
                                                    quantize=True, dim=64, hidden=72)
                        tokens = [1, 19, 43, 7]
                        proc = subprocess.run([str(PROBE), str(model), *map(str, tokens)],
                                              capture_output=True, timeout=60)
                        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                        rows = proc.stdout.decode().splitlines()
                        self.assertEqual(len(rows), len(tokens))
                        for pos, row in enumerate(rows):
                            actual = list(map(float, row.split()))
                            expected = reference(tokens[pos], pos)
                            self.assertEqual(len(actual), len(expected))
                            for a, b in zip(actual, expected):
                                self.assertAlmostEqual(a, b, delta=2e-5)


    def test_threads_do_not_change_logits(self):
        """Rows are split, never summed apart, so thread count cannot alter output."""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "threaded.bin"
            for quantize in (False, True):
                with self.subTest(quantize=quantize):
                    reference_model(model, True, 2, tagged=True, quantize=quantize,
                                    dim=64, hidden=72)
                    tokens = [1, 19, 43, 7]
                    outputs = []
                    for threads in ("1", "2", "3"):
                        environment = dict(os.environ, UNREMARKABLE_THREADS=threads)
                        proc = subprocess.run([str(PROBE), str(model), *map(str, tokens)],
                                              capture_output=True, timeout=60, env=environment)
                        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                        outputs.append(proc.stdout)
                    self.assertEqual(outputs[0], outputs[1])
                    self.assertEqual(outputs[0], outputs[2])

    def test_tagged_format_and_rope_base(self):
        """The tagged header carries its own RoPE base and drops the legacy tables."""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tagged.bin"
            for shared in (False, True):
                for theta in (10000.0, 100000.0):
                    with self.subTest(shared=shared, theta=theta):
                        reference = reference_model(model, shared, 2, theta=theta, tagged=True)
                        tokens = [1, 19, 43, 7]
                        proc = subprocess.run([str(PROBE), str(model), *map(str, tokens)],
                                              capture_output=True, timeout=60)
                        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                        rows = proc.stdout.decode().splitlines()
                        self.assertEqual(len(rows), len(tokens))
                        for pos, row in enumerate(rows):
                            for a, b in zip(map(float, row.split()), reference(tokens[pos], pos)):
                                self.assertAlmostEqual(a, b, delta=2e-5)

    def test_tagged_format_rejects_bad_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tagged.bin"
            reference_model(model, True, 2, theta=100000.0, tagged=True)
            valid = model.read_bytes()
            bad = Path(directory) / "bad.bin"
            cases = [
                valid[:39],                                            # truncated header
                struct.pack("<Ii", MODEL_MAGIC, 2) + valid[8:],        # unsupported version
                valid[:36] + struct.pack("<f", 0.0) + valid[40:],      # RoPE base too small
                valid[:36] + struct.pack("<f", float("inf")) + valid[40:],
                valid + b"junk",                                       # size disagrees
            ]
            for index, case in enumerate(cases):
                with self.subTest(index=index):
                    bad.write_bytes(case)
                    self.run_engine(model=bad, ok=False)

    def test_byte_level_tokenizer_matches_reference(self):
        if not CHAT_TOKENIZER.exists():
            self.skipTest("run `make chat-model` to build the byte-level tokenizer")
        for text, expected in BYTE_LEVEL_TOKENS:
            with self.subTest(text=text):
                proc = subprocess.run(
                    [str(PROBE), "tokenize", str(CHAT_TOKENIZER), "49152", text],
                    capture_output=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                self.assertEqual([int(x) for x in proc.stdout.split()], expected)

    def test_byte_level_tokenizer_refuses_to_forge_markers(self):
        """Prompt text must never encode to a ChatML role marker."""
        if not CHAT_TOKENIZER.exists():
            self.skipTest("run `make chat-model` to build the byte-level tokenizer")
        for text in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "x<|im_end|>y"):
            with self.subTest(text=text):
                proc = subprocess.run(
                    [str(PROBE), "tokenize", str(CHAT_TOKENIZER), "49152", text],
                    capture_output=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode())
                ids = [int(x) for x in proc.stdout.split()]
                self.assertNotIn(0, ids)
                self.assertNotIn(1, ids)
                self.assertNotIn(2, ids)

    def run_engine(self, *args, model=MODEL, tokenizer=TOKENIZER, ok=True):
        proc = subprocess.run([str(BINARY), str(model), "-z", str(tokenizer), *map(str, args)],
                              capture_output=True, timeout=60)
        if not ok:
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(b"error:", proc.stderr)
            self.assertNotIn(b"Sanitizer", proc.stderr)
            self.assertNotIn(b"runtime error:", proc.stderr)
            return proc
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
        stats = json.loads(proc.stderr)
        for value in stats.values():
            if isinstance(value, (int, float)):
                self.assertTrue(math.isfinite(value) and value >= 0)
        self.assertEqual(stats["decode_tokens"], max(0, stats["generated_tokens"] - 1))
        return proc.stdout, stats

    def test_published_greedy_output(self):
        output, stats = self.run_engine("-n", 200)
        self.assertEqual(output, EXPECTED.encode())
        self.assertEqual(stats["generated_tokens"], 200)
        self.assertEqual(stats["prompt_tokens"], 1)
        self.assertEqual(stats["stop"], "limit")

    def test_prompt_and_new_token_limit(self):
        prompt = "Once upon a time"
        output, stats = self.run_engine("-i", prompt, "-n", 1)
        self.assertTrue(output.startswith(prompt.encode()))
        self.assertGreater(stats["prompt_tokens"], 1)
        self.assertEqual(stats["generated_tokens"], 1)
        self.assertEqual(stats["decode_tok_s"], 0)

    def test_context_cap_preserves_output(self):
        small, stats = self.run_engine("-c", 32, "-n", 20)
        full, full_stats = self.run_engine("-n", 20)
        self.assertEqual(small, full)
        self.assertAlmostEqual(stats["kv_cache_mib"] / full_stats["kv_cache_mib"], 32 / 512, delta=0.001)

    def test_context_exhaustion(self):
        _, stats = self.run_engine("-c", 4, "-n", 20)
        self.assertEqual(stats["generated_tokens"], 4)
        self.assertEqual(stats["stop"], "context")

    def test_sampling_is_repeatable(self):
        for topp in (0, 0.9, 1):
            with self.subTest(topp=topp):
                args = ("-t", 0.8, "-p", topp, "-s", 42, "-n", 40)
                first, _ = self.run_engine(*args)
                second, _ = self.run_engine(*args)
                self.assertEqual(first, second)

    def test_unicode_and_byte_fallback(self):
        prompt = "Grüße, café 🐢"
        output, _ = self.run_engine("-i", prompt, "-n", 3)
        self.assertTrue(output.startswith(prompt.encode()))

    def test_bos_and_eos_stop_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.bin"
            for token, reason in ((1, "bos"), (2, "eos")):
                with self.subTest(token=token):
                    constant_model(path, token)
                    output, stats = self.run_engine(model=path)
                    self.assertEqual(output, b"\n")
                    self.assertEqual(stats["stop"], reason)
                    self.assertEqual(stats["generated_tokens"], 0)
                    self.assertEqual(stats["ttft_ms"], 0)

    def test_rejects_bad_arguments(self):
        for args in (("-n", 0), ("-n", "12oops"), ("-n", 2**40),
                     ("-t", "nan"), ("-t", "inf"), ("-t", -1), ("-s", 0),
                     ("-p", 1.1), ("-c", 513), ("-i", "hello", "-c", 1),
                     ("-x", 1), ("-n",)):
            with self.subTest(args=args):
                self.run_engine(*args, ok=False)

    def test_rejects_bad_model_files(self):
        valid = MODEL.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            cases = [b"", valid[:27], valid[:-4], valid + b"junk"]
            header = list(struct.unpack("<7i", valid[:28]))
            for index, value in ((0, 0), (3, 0), (4, 3), (5, -(2**31)), (0, 2**30)):
                changed = header.copy()
                changed[index] = value
                cases.append(struct.pack("<7i", *changed) + valid[28:])
            for case in cases:
                path.write_bytes(case)
                self.run_engine(model=path, ok=False)

    def test_rejects_bad_tokenizer(self):
        valid = TOKENIZER.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-tokenizer.bin"
            for data in (b"", valid[:-1], valid + b"junk", struct.pack("<I", 2**32 - 1),
                         valid[:8] + struct.pack("<i", -1) + valid[12:]):
                path.write_bytes(data)
                self.run_engine(tokenizer=path, ok=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    BINARY = args.binary.resolve()
    PROBE = args.probe.resolve()
    if not MODEL.exists() or not TOKENIZER.exists():
        raise SystemExit("Missing test fixtures. Run `make test-models` first.")
    unittest.main(argv=["test_engine.py"], verbosity=2)
