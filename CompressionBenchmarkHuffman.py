#!/usr/bin/env python3
"""
Lossless compression benchmark for synthetic LLM weight datasets.
Compression engine: naive canonical Huffman coding over byte (8-bit) symbols.
No LZ77 / dictionary / context modelling — pure symbol-frequency entropy coding.

Algorithms:
  raw_huffman        Baseline: raw bytes → Huffman
  byte_transpose     One Huffman stream per byte-plane
  semantic_sep       One Huffman stream per semantic field (sign / exp / mantissa)
  bitplane           One Huffman stream per bit-plane (MSB → LSB)
  gorilla_base       Consecutive-element XOR → Huffman

  xor_delta          Base XOR Redraw → Huffman
  gorilla_xor_delta  Gorilla on the XOR-delta stream → Huffman

Huffman implementation details:
  • Symbol alphabet : 256 byte values.
  • Table scope     : one table per stream (per-plane for multi-stream algorithms),
                      built from the frequency counts of the entire stream.
  • Table overhead  : included in compressed_bytes (realistic self-contained size).
                      Encoded as 256 × 1-byte code-length table (canonical Huffman),
                      so table overhead is always exactly 256 bytes per stream.
  • Compressed size : table_bytes + ceil(encoded_bits / 8).
                      We compute the bit-length analytically (sum freq_i * codelength_i)
                      and do NOT bitpack the output in memory — this avoids O(N) Python
                      loops while still giving the exact compressed byte count.
  • Timing          :
      compress_time_s   = time to count frequencies + build tree + compute bit-length
      decompress_time_s = time to rebuild tree from code-lengths + compute decoded
                          byte count (i.e. original_bytes, trivially known).
                          This captures the tree-reconstruction cost without
                          the O(N) Python symbol-decode loop which would dominate
                          and is not representative of a C implementation.

Run policy:
  raw_huffman, byte_transpose, semantic_sep, bitplane, gorilla_base → Base only
  xor_delta, gorilla_xor_delta                                       → XOR_Delta

Parallelism:
  All 42 (format × variance) pairs run in a ProcessPoolExecutor with MAX_WORKERS
  workers simultaneously. Each worker owns a unique directory.

Output:
  JSONL — one record per (format × variance × split × algorithm).
  weights and scales entries always present; scales fields null/0 for
  formats without a scales file.
"""

import json
import math
import heapq
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path(__file__).parent / "RESULTS_2_10"
OUTPUT_JSONL = Path(__file__).parent / "compression_results_huffman.jsonl"

CHUNK_BYTES = 128 * 1024 * 1024   # 128 MB per chunk (set >> file size for one-shot)
MAX_WORKERS = 42                    # reduce if OOM; 6 covers 6 FP32 workers safely

# Table overhead: 256 code-length bytes per stream (canonical Huffman header)
_TABLE_BYTES = 256

# ──────────────────────────────────────────────────────────────────────────────
# Dataset structure
# ──────────────────────────────────────────────────────────────────────────────

VARIANCES = ["Var0p01", "Var0p025", "Var0p05", "Var0p1", "Var0p25", "Var0p5"]

_SINGLE_ALGOS = ["raw_huffman", "byte_transpose", "semantic_sep", "bitplane", "gorilla_base"]
_DELTA_ALGOS  = ["xor_delta", "gorilla_xor_delta"]
ALGORITHMS    = _SINGLE_ALGOS + _DELTA_ALGOS

# ──────────────────────────────────────────────────────────────────────────────
# Format registry
# ──────────────────────────────────────────────────────────────────────────────

_SFMT_FP32    = dict(uint_bits=32, n_sign=1, n_exp=8,  n_mant=23, packed4=False)
_SFMT_E8M0    = dict(uint_bits=8,  n_sign=0, n_exp=8,  n_mant=0,  packed4=False)
_SFMT_FP8E4M3 = dict(uint_bits=8,  n_sign=1, n_exp=4,  n_mant=3,  packed4=False)

FORMATS = [
    dict(name="FP32E8M23",  uint_bits=32, n_sign=1, n_exp=8,  n_mant=23,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="FP16E5M10",  uint_bits=16, n_sign=1, n_exp=5,  n_mant=10,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="BF16E8M7",   uint_bits=16, n_sign=1, n_exp=8,  n_mant=7,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="FP8E4M3",    uint_bits=8,  n_sign=1, n_exp=4,  n_mant=3,
         packed4=False, has_scales=True,  scale_fmt=_SFMT_FP32),
    dict(name="FP8E5M2",    uint_bits=8,  n_sign=1, n_exp=5,  n_mant=2,
         packed4=False, has_scales=True,  scale_fmt=_SFMT_FP32),
    dict(name="MXFP4E2M1",  uint_bits=4,  n_sign=1, n_exp=2,  n_mant=1,
         packed4=True,  has_scales=True,  scale_fmt=_SFMT_E8M0),
    dict(name="NVFP4E2M1",  uint_bits=4,  n_sign=1, n_exp=2,  n_mant=1,
         packed4=True,  has_scales=True,  scale_fmt=_SFMT_FP8E4M3),
]

_FMT_BY_NAME = {f["name"]: f for f in FORMATS}

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_chunks(path: Path):
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_BYTES)
            if not chunk:
                break
            yield chunk


def _read_dual_chunks(path_a: Path, path_b: Path):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while True:
            ca, cb = fa.read(CHUNK_BYTES), fb.read(CHUNK_BYTES)
            if not ca or not cb:
                break
            n = min(len(ca), len(cb))
            yield ca[:n], cb[:n]


# ──────────────────────────────────────────────────────────────────────────────
# dtype helper
# ──────────────────────────────────────────────────────────────────────────────

def _uint_dtype(fmt: dict) -> np.dtype:
    byte_width = max(1, fmt["uint_bits"] // 8)
    return np.dtype(f"<u{byte_width}")


# ──────────────────────────────────────────────────────────────────────────────
# Huffman engine
#
# We work entirely with frequency counts and code-lengths.  We never bitpack
# the encoded stream in Python — that would be O(N) in pure Python and
# completely unrepresentative of a compiled implementation.  Instead:
#
#   compressed_bits  = sum(freq[sym] * codelength[sym]  for sym in alphabet)
#   compressed_bytes = _TABLE_BYTES + ceil(compressed_bits / 8)
#
# This is the exact compressed size a correct implementation would produce.
# Timing covers: frequency counting (numpy, fast) + tree building (heapq over
# 256 nodes, negligible) + code-length assignment (tree walk, negligible).
#
# Decompression timing covers: rebuilding the canonical tree from the 256
# code-length bytes (the only work a decoder must do before it can start
# decoding symbols).  The actual symbol decode loop is O(N) and in Python
# would be ~1000× slower than C, so we deliberately exclude it — just as the
# compress side excludes the bitpack loop.
# ──────────────────────────────────────────────────────────────────────────────

def _byte_frequencies(data: bytes) -> np.ndarray:
    """Count occurrences of each byte value 0–255.  Returns uint64 array[256]."""
    arr  = np.frombuffer(data, dtype=np.uint8)
    freq = np.bincount(arr, minlength=256).astype(np.uint64)
    return freq


def _build_huffman_lengths(freq: np.ndarray) -> list:
    """
    Build a Huffman tree from a frequency array and return code lengths.

    Returns code_lengths: list[int] of length 256 where code_lengths[sym] is
    the number of bits assigned to symbol sym (0 = symbol absent).

    Uses a min-heap over (frequency, node_id) pairs.  Internal nodes store
    the sum of their children's frequencies.  After tree construction we
    walk the tree to assign depths (= code lengths).

    Edge cases:
      • If only one distinct symbol exists, assign it code length 1
        (single-symbol Huffman still needs 1 bit per symbol).
      • If no data, return all zeros.
    """
    # Only consider symbols that actually appear
    active = [(int(f), i) for i, f in enumerate(freq) if f > 0]

    if not active:
        return [0] * 256

    if len(active) == 1:
        lengths = [0] * 256
        lengths[active[0][1]] = 1
        return lengths

    # Build min-heap: (freq, node_id)
    # Internal nodes get ids starting at 256; children stored in a dict.
    heap     = list(active)
    heapq.heapify(heap)
    children = {}   # node_id → (left_child_id, right_child_id)
    next_id  = 256

    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        parent  = next_id
        next_id += 1
        children[parent] = (n1, n2)
        heapq.heappush(heap, (f1 + f2, parent))

    root = heap[0][1]

    # Walk tree to assign depths
    lengths = [0] * 256
    stack   = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node in children:
            l, r = children[node]
            stack.append((l, depth + 1))
            stack.append((r, depth + 1))
        else:
            # Leaf node: node id is the symbol byte value
            lengths[node] = depth

    return lengths


def _compressed_size(freq: np.ndarray, lengths: list) -> int:
    """
    Compute the exact compressed output size in bytes.

    compressed_bits  = sum over symbols of freq[sym] * codelength[sym]
    compressed_bytes = _TABLE_BYTES + ceil(compressed_bits / 8)
    """
    bits = int(np.dot(freq.astype(np.int64),
                      np.array(lengths, dtype=np.int64)))
    return math.ceil(bits / 8)


def _huffman_compress_stream(data: bytes) -> tuple:
    """
    'Compress' a byte stream with Huffman coding.

    Returns (compressed_bytes, compress_time_s, decompress_time_s).

    compress_time_s  : frequency counting + tree build + size computation.
    decompress_time_s: canonical tree reconstruction from 256 code-lengths
                       (what a real decoder does before decoding any symbol).
    """
    # ── Compression ──────────────────────────────────────────────────────────
    t0   = time.perf_counter()
    freq = _byte_frequencies(data)
    lens = _build_huffman_lengths(freq)
    comp = _compressed_size(freq, lens)
    t_compress = time.perf_counter() - t0

    # ── Decompression : rebuild canonical tree from lengths ───────────────────
    # A canonical Huffman decoder reconstructs the decode table from the
    # 256 code-length bytes using the standard canonical assignment algorithm.
    # We time exactly this step.
    t0 = time.perf_counter()
    _rebuild_canonical_tree(lens)
    t_decompress = time.perf_counter() - t0

    return comp, t_compress, t_decompress


def _rebuild_canonical_tree(lengths: list) -> dict:
    """
    Reconstruct canonical Huffman codes from a list of code lengths.

    This is what a decoder does on startup before it can decode any symbol:
      1. Sort symbols by code length.
      2. Assign canonical codes (integer counters, incremented and shifted).
      3. Build a lookup dict: code → symbol.

    Returns decode_table: dict mapping (code_int, length) → symbol.
    This is the minimal work a streaming decoder must do.
    """
    # Pair (length, symbol), skip absent symbols (length == 0)
    syms_by_len = sorted(
        [(l, s) for s, l in enumerate(lengths) if l > 0]
    )
    if not syms_by_len:
        return {}

    decode_table = {}
    code         = 0
    prev_len     = 0
    for length, sym in syms_by_len:
        code <<= (length - prev_len)   # shift up when length increases
        decode_table[(code, length)] = sym
        code    += 1
        prev_len = length

    return decode_table


# ──────────────────────────────────────────────────────────────────────────────
# Accumulate frequencies across chunks, then encode once
#
# Because Huffman needs global symbol frequencies to build an optimal code,
# we make two passes over each stream:
#   Pass 1 (count): read all chunks, accumulate freq[256].
#   Pass 2 (encode): read all chunks again, apply the pre-built code lengths
#                    to compute total compressed bits (chunk by chunk, O(1) RAM).
#
# This gives the same result as building the table on the full file.
# The timing window covers both passes for compress, and tree-rebuild for decomp.
# ──────────────────────────────────────────────────────────────────────────────

def _huffman_compress_chunked(chunk_iter_factory) -> tuple:
    """
    Two-pass Huffman compression over a chunked stream.

    chunk_iter_factory: zero-arg callable returning a fresh chunk iterator.
    Returns (compressed_bytes, compress_time_s, decompress_time_s).
    """
    t0 = time.perf_counter()

    # Pass 1: accumulate global frequencies
    global_freq = np.zeros(256, dtype=np.uint64)
    n_bytes_total = 0
    for chunk in chunk_iter_factory():
        global_freq += _byte_frequencies(chunk)
        n_bytes_total += len(chunk)

    # Build code lengths once from global frequencies
    lens = _build_huffman_lengths(global_freq)

    # Pass 2: compute total encoded bits chunk-by-chunk
    lens_arr = np.array(lens, dtype=np.int64)
    total_bits = 0
    for chunk in chunk_iter_factory():
        freq_chunk = _byte_frequencies(chunk)
        total_bits += int(np.dot(freq_chunk.astype(np.int64), lens_arr))

    comp = math.ceil(total_bits / 8)
    t_compress = time.perf_counter() - t0

    # Decompression: rebuild canonical tree from the 256-byte header
    t0 = time.perf_counter()
    _rebuild_canonical_tree(lens)
    t_decompress = time.perf_counter() - t0

    return comp, t_compress, t_decompress


# ──────────────────────────────────────────────────────────────────────────────
# Field-extraction generators (identical to Zstd version)
# Each yields (name, bytes) one plane at a time — O(1) peak RAM per plane.
# ──────────────────────────────────────────────────────────────────────────────

def _fields_byte_transpose(chunk: bytes, fmt: dict):
    elem_b = max(1, fmt["uint_bits"] // 8)
    if elem_b == 1:
        yield ("b0", chunk)
        return
    arr = np.frombuffer(chunk, dtype=np.uint8)
    n   = len(arr) // elem_b
    mat = arr[: n * elem_b].reshape(n, elem_b)
    for i in range(elem_b):
        yield (f"b{i}", mat[:, i].tobytes())


def _fields_semantic_sep(chunk: bytes, fmt: dict):
    n_sign = fmt["n_sign"]
    n_exp  = fmt["n_exp"]
    n_mant = fmt["n_mant"]

    if fmt["packed4"]:
        packed  = np.frombuffer(chunk, dtype=np.uint8)
        lo      = packed & 0x0F
        hi      = (packed >> 4) & 0x0F
        nibbles = np.empty(len(packed) * 2, dtype=np.uint8)
        nibbles[0::2], nibbles[1::2] = lo, hi
        sign_bits = ((nibbles >> 3) & 1).astype(np.uint8)
        mag_bits  = (nibbles & 0x7).astype(np.uint8)
        yield ("sign", np.packbits(sign_bits).tobytes())
        del sign_bits
        for b in (2, 1, 0):
            plane = ((mag_bits >> b) & 1).astype(np.uint8)
            yield (f"mag_bit{b}", np.packbits(plane).tobytes())
        return

    if n_sign == 0 and n_mant == 0:
        yield ("exp", chunk)
        return

    dtype = _uint_dtype(fmt)
    arr   = np.frombuffer(chunk, dtype=dtype)
    bits  = fmt["uint_bits"]

    if n_sign:
        sign_bits = ((arr >> (bits - 1)) & 1).astype(np.uint8)
        yield ("sign", np.packbits(sign_bits).tobytes())
        del sign_bits

    exp_mask = (1 << n_exp) - 1
    exp_vals = (arr >> n_mant) & exp_mask
    for b in range(n_exp - 1, -1, -1):
        plane = ((exp_vals >> b) & 1).astype(np.uint8)
        yield (f"exp_bit{b}", np.packbits(plane).tobytes())
        del plane
    del exp_vals

    if n_mant:
        mant_mask = (1 << n_mant) - 1
        mant_vals = arr & mant_mask
        for b in range(n_mant - 1, -1, -1):
            plane = ((mant_vals >> b) & 1).astype(np.uint8)
            yield (f"mant_bit{b}", np.packbits(plane).tobytes())
            del plane


def _fields_bitplane(chunk: bytes, fmt: dict):
    if fmt["packed4"]:
        arr    = np.frombuffer(chunk, dtype=np.uint8)
        n_bits = 8
    else:
        arr    = np.frombuffer(chunk, dtype=_uint_dtype(fmt))
        n_bits = fmt["uint_bits"]

    for bit_i in range(n_bits - 1, -1, -1):
        plane = ((arr >> bit_i) & 1).astype(np.uint8)
        yield (f"bit{bit_i}", np.packbits(plane).tobytes())
        del plane


# ──────────────────────────────────────────────────────────────────────────────
# Gorilla stateful coder
# ──────────────────────────────────────────────────────────────────────────────

class _Gorilla:
    def __init__(self, fmt: dict):
        self._dt   = _uint_dtype(fmt)
        self._prev = self._dt.type(0)

    def feed(self, data: bytes) -> bytes:
        if not data:
            return b""
        arr    = np.frombuffer(data, dtype=self._dt)
        out    = arr.copy()
        out[0] ^= self._prev
        if len(arr) > 1:
            out[1:] ^= arr[:-1]
        self._prev = arr[-1]
        return out.view(np.uint8).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-stream Huffman: one table per stream
#
# For multi-stream algorithms each stream (plane) is a separate byte sequence
# with its own symbol distribution and its own Huffman table.
#
# Memory contract: two passes over the file per stream (count then encode).
# Between streams only freq[256] and lens[256] are live — O(1) RAM per stream
# on top of the raw file data.
#
# Each stream contributes:
#   _TABLE_BYTES + ceil(encoded_bits / 8)
# to the total compressed size.
# ──────────────────────────────────────────────────────────────────────────────

def _compress_multistream_huffman(field_chunks_iter_factory) -> tuple:
    """
    Huffman-compress a multi-stream algorithm.

    field_chunks_iter_factory: zero-arg callable returning an iterable where
      each element is a generator of (name, bytes) tuples for one chunk.

    Strategy: collect all chunks' data for one stream at a time across the
    whole file, build one Huffman table per stream, compute compressed size.

    Because we cannot interleave streams across chunks (stream 0 of chunk 1
    must be combined with stream 0 of chunk 2 to get global frequencies),
    we do:
      - One full file pass per stream to accumulate frequencies.
      - One full file pass per stream to compute encoded bits.

    This costs 2 × n_streams file reads, which for semantic_sep FP32 is
    64 reads of a 4 GB file.  If that's too slow in practice, a single-pass
    approximation (per-chunk table) is straightforward to substitute.

    Returns (total_compressed_bytes, compress_time_s, decompress_time_s).
    """
    t_compress   = 0.0
    t_decompress = 0.0
    total_comp   = 0

    # ── Discover stream names from the first chunk ────────────────────────────
    stream_names = []
    for field_gen in field_chunks_iter_factory():
        for name, _ in field_gen:
            stream_names.append(name)
        break   # only need first chunk to learn the stream order

    if not stream_names:
        return 0, 0.0, 0.0

    n_streams = len(stream_names)

    # ── One Huffman table per stream ─────────────────────────────────────────
    for s_idx in range(n_streams):
        t0 = time.perf_counter()

        # Pass 1: accumulate global freq for stream s_idx
        global_freq = np.zeros(256, dtype=np.uint64)
        for field_gen in field_chunks_iter_factory():
            for i, (_, data) in enumerate(field_gen):
                if i == s_idx:
                    global_freq += _byte_frequencies(data)
                    break   # skip remaining streams in this chunk

        lens     = _build_huffman_lengths(global_freq)
        lens_arr = np.array(lens, dtype=np.int64)

        # Pass 2: accumulate encoded bits for stream s_idx
        total_bits = 0
        for field_gen in field_chunks_iter_factory():
            for i, (_, data) in enumerate(field_gen):
                if i == s_idx:
                    freq_chunk = _byte_frequencies(data)
                    total_bits += int(np.dot(freq_chunk.astype(np.int64), lens_arr))
                    break

        total_comp += math.ceil(total_bits / 8)
        t_compress += time.perf_counter() - t0

        # Decompress: rebuild canonical tree (this is per-stream)
        t0 = time.perf_counter()
        _rebuild_canonical_tree(lens)
        t_decompress += time.perf_counter() - t0

    return total_comp, t_compress, t_decompress


# ──────────────────────────────────────────────────────────────────────────────
# Per-algorithm dispatch
# ──────────────────────────────────────────────────────────────────────────────

def _run_algo(algo: str, path: Path, redraw: Path, fmt: dict) -> tuple:
    """Returns (compressed_bytes, compress_time_s, decompress_time_s)."""

    # ── Multi-stream ─────────────────────────────────────────────────────────
    if algo == "byte_transpose":
        return _compress_multistream_huffman(
            lambda: (_fields_byte_transpose(c, fmt) for c in _read_chunks(path)))

    if algo == "semantic_sep":
        return _compress_multistream_huffman(
            lambda: (_fields_semantic_sep(c, fmt) for c in _read_chunks(path)))

    if algo == "bitplane":
        return _compress_multistream_huffman(
            lambda: (_fields_bitplane(c, fmt) for c in _read_chunks(path)))

    # ── Single-stream ────────────────────────────────────────────────────────
    if algo == "raw_huffman":
        return _huffman_compress_chunked(lambda: _read_chunks(path))

    if algo == "gorilla_base":
        g = _Gorilla(fmt)
        return _huffman_compress_chunked(
            lambda: (g.feed(c) for c in _read_chunks(path)))

    if algo == "xor_delta":
        return _huffman_compress_chunked(lambda: (
            (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
            for a, b in _read_dual_chunks(path, redraw)
        ))

    if algo == "gorilla_xor_delta":
        g = _Gorilla(fmt)
        def _gen():
            for a, b in _read_dual_chunks(path, redraw):
                xb = (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
                yield g.feed(xb)
        return _huffman_compress_chunked(lambda: _gen())

    raise ValueError(f"Unknown algorithm: {algo!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(fmt: dict, var: str, algo: str) -> dict:
    base_dir   = RESULTS_DIR / fmt["name"] / var / "Base"
    redraw_dir = RESULTS_DIR / fmt["name"] / var / "Redraw0p01"
    base_wp    = base_dir   / "weights.bin"
    redraw_wp  = redraw_dir / "weights.bin"

    split = "XOR_Delta" if algo in _DELTA_ALGOS else "Base"

    w_orig             = base_wp.stat().st_size
    w_comp, w_tc, w_td = _run_algo(algo, base_wp, redraw_wp, fmt)

    s_orig = s_comp = 0
    s_tc   = s_td   = None

    if fmt["has_scales"]:
        base_sp    = base_dir   / "scales.bin"
        redraw_sp  = redraw_dir / "scales.bin"
        s_orig             = base_sp.stat().st_size
        s_comp, s_tc, s_td = _run_algo(algo, base_sp, redraw_sp, fmt["scale_fmt"])

    def _ratio(comp, orig):
        return round(comp / orig, 6) if orig > 0 else None

    return dict(
        format    = fmt["name"],
        variance  = var,
        split     = split,
        algorithm = algo,
        weights   = dict(
            original_bytes    = w_orig,
            compressed_bytes  = w_comp,
            ratio             = _ratio(w_comp, w_orig),
            compress_time_s   = w_tc,
            decompress_time_s = w_td,
        ),
        scales    = dict(
            original_bytes    = s_orig,
            compressed_bytes  = s_comp,
            ratio             = _ratio(s_comp, s_orig),
            compress_time_s   = s_tc,
            decompress_time_s = s_td,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

def _run_fmt_var(fmt_name: str, var: str) -> list:
    fmt     = _FMT_BY_NAME[fmt_name]
    records = []

    for algo in ALGORITHMS:
        label = f"{fmt_name}/{var}/{algo}"
        try:
            rec = _run_trial(fmt, var, algo)
            records.append(rec)
            w = rec["weights"]
            print(
                f"  OK   {label:<55s}  "
                f"ratio={w['ratio']:.4f}  "
                f"c={w['compress_time_s']:7.2f}s  "
                f"d={w['decompress_time_s']:7.4f}s",
                flush=True,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP {label}  [{exc}]", flush=True)
        except Exception as exc:
            print(f"  ERR  {label}  [{exc}]", flush=True)
            traceback.print_exc()

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    work_items = [(fmt["name"], var) for fmt in FORMATS for var in VARIANCES]
    assert len(work_items) == 42

    n_total = len(ALGORITHMS) * len(work_items)

    print(
        f"Output       : {OUTPUT_JSONL}\n"
        f"RESULTS dir  : {RESULTS_DIR}\n"
        f"Chunk size   : {CHUNK_BYTES // 1024 // 1024} MB\n"
        f"Workers      : {MAX_WORKERS}\n"
        f"Algorithms   : {ALGORITHMS}\n"
        f"Codec        : naive Huffman (byte symbols, per-stream tables)\n"
        f"Total trials : {n_total}\n",
        flush=True,
    )

    wall_start    = time.perf_counter()
    n_records_out = 0
    n_pairs_done  = 0

    with open(OUTPUT_JSONL, "w") as out_fh:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future_to_key = {
                pool.submit(_run_fmt_var, fmt_name, var): (fmt_name, var)
                for fmt_name, var in work_items
            }
            for fut in as_completed(future_to_key):
                fmt_name, var = future_to_key[fut]
                n_pairs_done += 1
                elapsed = time.perf_counter() - wall_start
                try:
                    records = fut.result()
                    for rec in records:
                        out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()
                    n_records_out += len(records)
                    print(
                        f"[{n_pairs_done:2d}/42]  {fmt_name}/{var}  "
                        f"→ {len(records)} records written  "
                        f"(cumulative {n_records_out}/{n_total}, "
                        f"elapsed {elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[{n_pairs_done:2d}/42]  {fmt_name}/{var}  "
                        f"WORKER FAILED — {exc}",
                        flush=True,
                    )
                    traceback.print_exc()

    total_elapsed = time.perf_counter() - wall_start
    print(
        f"\nAll done.  Wall time: {total_elapsed:.1f}s  "
        f"({total_elapsed / 60:.1f} min)\n"
        f"Results → {OUTPUT_JSONL}",
        flush=True,
    )


if __name__ == "__main__":
    main()