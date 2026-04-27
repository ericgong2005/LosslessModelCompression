#!/usr/bin/env python3
"""
Lossless compression benchmark for synthetic LLM weight datasets.
Compression engine: naive canonical Huffman coding over byte (8-bit) symbols.

Algorithms:
  raw_huffman        Baseline: raw bytes → Huffman  (single stream)
  byte_transpose     One Huffman stream per byte-plane
  semantic_sep       One Huffman stream per semantic field (sign / exp / mantissa)
  bitplane           One Huffman stream per bit-plane (MSB → LSB)
  gorilla_base       Consecutive-element XOR → Huffman  (single stream)
  xor_delta          Base XOR Redraw → Huffman  (single stream)
  gorilla_xor_delta  Gorilla on the XOR-delta stream → Huffman  (single stream)

Output schema per record:
  format, variance, split, algorithm
  timing:
    compress_time_s    — total across all streams (weights + scales)
    decompress_time_s  — total across all streams (weights + scales)
  weights:
    original_bytes, compressed_bytes, ratio   — totals across all streams
    streams: { stream_name: { original_bytes, compressed_bytes, ratio,
                               compress_time_s, decompress_time_s } }
  scales:  (same structure; null fields when format has no scales)

For single-stream algorithms (raw_huffman, gorilla_base, xor_delta,
gorilla_xor_delta) weights.streams has one entry keyed "data".
For byte_transpose: entries keyed "b0", "b1", …
For semantic_sep:   entries keyed "sign", "exp_bit7", …, "mant_bit0", …
For bitplane:       entries keyed "bit31", "bit30", …, "bit0"

Parallelism:
  Work items are (format, variance, algorithm) triples — 42 × 7 = 294 total.
  They are submitted to the pool algorithm-by-algorithm so that faster
  algorithms drain before slower ones pile up, but we don't wait for all
  workers of one algorithm to finish before starting the next — as soon as
  a worker slot is free the next item in sequence is dispatched.
  MAX_WORKERS workers run concurrently; each worker owns a unique
  (format, variance) file pair for the duration of its single algorithm trial.
"""

import json
import math
import heapq
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path(__file__).parent / "RESULTS_2_30"
OUTPUT_JSONL = Path(__file__).parent / "compression_results_huffman_finegrained.jsonl"

CHUNK_BYTES = 128 * 1024 * 1024   # set larger than any file for one-shot reads
MAX_WORKERS = 42                    # safe for 69 GB node; raise if RAM allows

# ──────────────────────────────────────────────────────────────────────────────
# Dataset structure
# ──────────────────────────────────────────────────────────────────────────────

VARIANCES = ["Var0p01", "Var0p025", "Var0p05", "Var0p1", "Var0p25", "Var0p5"]

_DELTA_ALGOS  = {"xor_delta", "gorilla_xor_delta"}
ALGORITHMS    = ["raw_huffman", "byte_transpose", "semantic_sep", "bitplane",
                 "gorilla_base", "xor_delta", "gorilla_xor_delta"]

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
# ──────────────────────────────────────────────────────────────────────────────

def _byte_frequencies(data: bytes) -> np.ndarray:
    arr  = np.frombuffer(data, dtype=np.uint8)
    return np.bincount(arr, minlength=256).astype(np.uint64)


def _build_huffman_lengths(freq: np.ndarray) -> list:
    active = [(int(f), i) for i, f in enumerate(freq) if f > 0]
    if not active:
        return [0] * 256
    if len(active) == 1:
        lengths = [0] * 256
        lengths[active[0][1]] = 1
        return lengths

    heap     = list(active)
    heapq.heapify(heap)
    children = {}
    next_id  = 256

    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        parent  = next_id
        next_id += 1
        children[parent] = (n1, n2)
        heapq.heappush(heap, (f1 + f2, parent))

    root    = heap[0][1]
    lengths = [0] * 256
    stack   = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node in children:
            l, r = children[node]
            stack.append((l, depth + 1))
            stack.append((r, depth + 1))
        else:
            lengths[node] = depth
    return lengths


def _rebuild_canonical_tree(lengths: list) -> dict:
    syms_by_len = sorted([(l, s) for s, l in enumerate(lengths) if l > 0])
    if not syms_by_len:
        return {}
    decode_table = {}
    code = prev_len = 0
    for length, sym in syms_by_len:
        code <<= (length - prev_len)
        decode_table[(code, length)] = sym
        code    += 1
        prev_len = length
    return decode_table


def _compressed_bytes_from_bits(bits: int) -> int:
    return math.ceil(bits / 8)


# ──────────────────────────────────────────────────────────────────────────────
# Single-stream Huffman: returns a StreamResult dict
#
# StreamResult = {
#   "original_bytes":    int,
#   "compressed_bytes":  int,
#   "ratio":             float,
#   "compress_time_s":   float,
#   "decompress_time_s": float,
# }
# ──────────────────────────────────────────────────────────────────────────────

def _huffman_one_stream(chunk_iter_factory) -> dict:
    """
    Two-pass Huffman over a single byte stream.
    Pass 1: accumulate global frequencies.
    Pass 2: compute total encoded bits using global code lengths.
    """
    t0 = time.perf_counter()

    global_freq   = np.zeros(256, dtype=np.uint64)
    n_orig        = 0
    for chunk in chunk_iter_factory():
        global_freq += _byte_frequencies(chunk)
        n_orig      += len(chunk)

    lens     = _build_huffman_lengths(global_freq)
    lens_arr = np.array(lens, dtype=np.int64)

    total_bits = 0
    for chunk in chunk_iter_factory():
        total_bits += int(np.dot(_byte_frequencies(chunk).astype(np.int64), lens_arr))

    n_comp     = _compressed_bytes_from_bits(total_bits)
    t_compress = time.perf_counter() - t0

    t0 = time.perf_counter()
    _rebuild_canonical_tree(lens)
    t_decompress = time.perf_counter() - t0

    return {
        "original_bytes":    n_orig,
        "compressed_bytes":  n_comp,
        "ratio":             round(n_comp / n_orig, 6) if n_orig else None,
        "compress_time_s":   t_compress,
        "decompress_time_s": t_decompress,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-stream Huffman: returns dict of stream_name → StreamResult
# plus aggregate original_bytes, compressed_bytes, ratio
# ──────────────────────────────────────────────────────────────────────────────

def _huffman_multi_stream(field_chunks_iter_factory) -> dict:
    """
    Per-stream two-pass Huffman.

    field_chunks_iter_factory: zero-arg callable → iterable of generators,
      each generator yielding (name, bytes) for one chunk's streams.

    Returns:
      {
        "original_bytes":   int,   # sum across streams
        "compressed_bytes": int,   # sum across streams
        "ratio":            float,
        "streams": {
          stream_name: StreamResult,
          ...
        }
      }

    For each stream s_idx we do:
      Pass 1: accumulate global_freq by consuming only stream s_idx from
              each chunk (advance-and-skip the generator).
      Pass 2: compute encoded bits the same way.

    This costs 2 × n_streams full file reads.  Peak RAM = one stream's data
    at a time (generator yields one plane, we consume it, then discard).
    """
    # ── Discover stream names from the first chunk ────────────────────────────
    stream_names = []
    for field_gen in field_chunks_iter_factory():
        for name, _ in field_gen:
            stream_names.append(name)
        break
    if not stream_names:
        return {"original_bytes": 0, "compressed_bytes": 0, "ratio": None,
                "streams": {}}

    n_streams    = len(stream_names)
    stream_results = {}

    for s_idx, s_name in enumerate(stream_names):
        t0 = time.perf_counter()

        # Pass 1: global frequencies for stream s_idx
        global_freq = np.zeros(256, dtype=np.uint64)
        n_orig      = 0
        for field_gen in field_chunks_iter_factory():
            for i, (_, data) in enumerate(field_gen):
                if i == s_idx:
                    global_freq += _byte_frequencies(data)
                    n_orig      += len(data)
                    break   # skip remaining streams in this chunk

        lens     = _build_huffman_lengths(global_freq)
        lens_arr = np.array(lens, dtype=np.int64)

        # Pass 2: encoded bits for stream s_idx
        total_bits = 0
        for field_gen in field_chunks_iter_factory():
            for i, (_, data) in enumerate(field_gen):
                if i == s_idx:
                    total_bits += int(
                        np.dot(_byte_frequencies(data).astype(np.int64), lens_arr)
                    )
                    break

        n_comp     = _compressed_bytes_from_bits(total_bits)
        t_compress = time.perf_counter() - t0

        t0 = time.perf_counter()
        _rebuild_canonical_tree(lens)
        t_decompress = time.perf_counter() - t0

        stream_results[s_name] = {
            "original_bytes":    n_orig,
            "compressed_bytes":  n_comp,
            "ratio":             round(n_comp / n_orig, 6) if n_orig else None,
            "compress_time_s":   t_compress,
            "decompress_time_s": t_decompress,
        }

    total_orig = sum(v["original_bytes"]   for v in stream_results.values())
    total_comp = sum(v["compressed_bytes"] for v in stream_results.values())

    return {
        "original_bytes":   total_orig,
        "compressed_bytes": total_comp,
        "ratio":            round(total_comp / total_orig, 6) if total_orig else None,
        "streams":          stream_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Field-extraction generators (one plane at a time — O(1) RAM per plane)
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
        yield ("sign", np.packbits(sign_bits).tobytes());  del sign_bits
        for b in (2, 1, 0):
            plane = ((mag_bits >> b) & 1).astype(np.uint8)
            yield (f"mag_bit{b}", np.packbits(plane).tobytes());  del plane
        return

    if n_sign == 0 and n_mant == 0:     # E8M0 — pure exponent byte
        yield ("exp", chunk)
        return

    dtype = _uint_dtype(fmt)
    arr   = np.frombuffer(chunk, dtype=dtype)
    bits  = fmt["uint_bits"]

    if n_sign:
        sign_bits = ((arr >> (bits - 1)) & 1).astype(np.uint8)
        yield ("sign", np.packbits(sign_bits).tobytes());  del sign_bits

    exp_mask = (1 << n_exp) - 1
    exp_vals = (arr >> n_mant) & exp_mask
    for b in range(n_exp - 1, -1, -1):
        plane = ((exp_vals >> b) & 1).astype(np.uint8)
        yield (f"exp_bit{b}", np.packbits(plane).tobytes());  del plane
    del exp_vals

    if n_mant:
        mant_mask = (1 << n_mant) - 1
        mant_vals = arr & mant_mask
        for b in range(n_mant - 1, -1, -1):
            plane = ((mant_vals >> b) & 1).astype(np.uint8)
            yield (f"mant_bit{b}", np.packbits(plane).tobytes());  del plane


def _fields_bitplane(chunk: bytes, fmt: dict):
    if fmt["packed4"]:
        arr    = np.frombuffer(chunk, dtype=np.uint8)
        n_bits = 8
    else:
        arr    = np.frombuffer(chunk, dtype=_uint_dtype(fmt))
        n_bits = fmt["uint_bits"]
    for bit_i in range(n_bits - 1, -1, -1):
        plane = ((arr >> bit_i) & 1).astype(np.uint8)
        yield (f"bit{bit_i}", np.packbits(plane).tobytes());  del plane


# ──────────────────────────────────────────────────────────────────────────────
# Gorilla
# ──────────────────────────────────────────────────────────────────────────────

class _Gorilla:
    def __init__(self, fmt: dict):
        self._dt   = _uint_dtype(fmt)
        self._prev = self._dt.type(0)

    def feed(self, data: bytes) -> bytes:
        if not data:
            return b""
        arr        = np.frombuffer(data, dtype=self._dt)
        out        = arr.copy()
        out[0]    ^= self._prev
        if len(arr) > 1:
            out[1:] ^= arr[:-1]
        self._prev = arr[-1]
        return out.view(np.uint8).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Per-file compression dispatcher
#
# Returns a "file_result" dict:
#   single-stream algos:
#     { "original_bytes", "compressed_bytes", "ratio",
#       "compress_time_s", "decompress_time_s",
#       "streams": { "data": StreamResult } }
#   multi-stream algos:
#     { "original_bytes", "compressed_bytes", "ratio",
#       "compress_time_s", "decompress_time_s",
#       "streams": { stream_name: StreamResult, ... } }
# ──────────────────────────────────────────────────────────────────────────────

def _compress_file(algo: str, path: Path, redraw: Path, fmt: dict) -> dict:

    # ── Multi-stream ──────────────────────────────────────────────────────────
    if algo == "byte_transpose":
        res = _huffman_multi_stream(
            lambda: (_fields_byte_transpose(c, fmt) for c in _read_chunks(path)))

    elif algo == "semantic_sep":
        res = _huffman_multi_stream(
            lambda: (_fields_semantic_sep(c, fmt) for c in _read_chunks(path)))

    elif algo == "bitplane":
        res = _huffman_multi_stream(
            lambda: (_fields_bitplane(c, fmt) for c in _read_chunks(path)))

    # ── Single-stream ─────────────────────────────────────────────────────────
    elif algo == "raw_huffman":
        sr  = _huffman_one_stream(lambda: _read_chunks(path))
        res = {**sr, "streams": {"data": sr}}

    elif algo == "gorilla_base":
        g   = _Gorilla(fmt)
        sr  = _huffman_one_stream(lambda: (g.feed(c) for c in _read_chunks(path)))
        res = {**sr, "streams": {"data": sr}}

    elif algo == "xor_delta":
        sr  = _huffman_one_stream(lambda: (
            (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
            for a, b in _read_dual_chunks(path, redraw)
        ))
        res = {**sr, "streams": {"data": sr}}

    elif algo == "gorilla_xor_delta":
        g = _Gorilla(fmt)
        def _gxd():
            for a, b in _read_dual_chunks(path, redraw):
                xb = (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
                yield g.feed(xb)
        sr  = _huffman_one_stream(lambda: _gxd())
        res = {**sr, "streams": {"data": sr}}

    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")

    # Attach aggregate timing (sum of per-stream timings = total work)
    res["compress_time_s"]   = sum(
        s["compress_time_s"]   for s in res["streams"].values())
    res["decompress_time_s"] = sum(
        s["decompress_time_s"] for s in res["streams"].values())

    return res


# ──────────────────────────────────────────────────────────────────────────────
# Single trial — one (format, variance, algorithm)
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(fmt_name: str, var: str, algo: str) -> dict:
    fmt        = _FMT_BY_NAME[fmt_name]
    base_dir   = RESULTS_DIR / fmt_name / var / "Base"
    redraw_dir = RESULTS_DIR / fmt_name / var / "Redraw0p01"
    base_wp    = base_dir   / "weights.bin"
    redraw_wp  = redraw_dir / "weights.bin"

    split = "XOR_Delta" if algo in _DELTA_ALGOS else "Base"

    w_res = _compress_file(algo, base_wp, redraw_wp, fmt)

    # Scales
    if fmt["has_scales"]:
        base_sp  = base_dir   / "scales.bin"
        redraw_sp = redraw_dir / "scales.bin"
        s_res = _compress_file(algo, base_sp, redraw_sp, fmt["scale_fmt"])
    else:
        s_res = {
            "original_bytes": 0, "compressed_bytes": 0, "ratio": None,
            "compress_time_s": None, "decompress_time_s": None,
            "streams": {},
        }

    # Top-level timing = weights + scales combined
    s_tc = s_res["compress_time_s"]
    s_td = s_res["decompress_time_s"]
    total_compress   = w_res["compress_time_s"]   + (s_tc or 0.0)
    total_decompress = w_res["decompress_time_s"] + (s_td or 0.0)

    return dict(
        format    = fmt_name,
        variance  = var,
        split     = split,
        algorithm = algo,
        timing    = dict(
            compress_time_s   = total_compress,
            decompress_time_s = total_decompress,
        ),
        weights   = dict(
            original_bytes    = w_res["original_bytes"],
            compressed_bytes  = w_res["compressed_bytes"],
            ratio             = w_res["ratio"],
            compress_time_s   = w_res["compress_time_s"],
            decompress_time_s = w_res["decompress_time_s"],
            streams           = w_res["streams"],
        ),
        scales    = dict(
            original_bytes    = s_res["original_bytes"],
            compressed_bytes  = s_res["compressed_bytes"],
            ratio             = s_res["ratio"],
            compress_time_s   = s_tc,
            decompress_time_s = s_td,
            streams           = s_res["streams"],
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Build work items ordered algorithm-first so that faster algorithms drain
    # before slower ones, minimising idle worker time.
    # Order: algorithm (outer) × format × variance (inner)
    work_items = [
        (fmt["name"], var, algo)
        for algo in ALGORITHMS
        for fmt  in FORMATS
        for var  in VARIANCES
    ]
    n_total = len(work_items)   # 7 × 7 × 6 = 294

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

    with open(OUTPUT_JSONL, "w") as out_fh:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # Submit all work items in algorithm-first order.
            # as_completed returns futures as they finish (any order),
            # but submission order is algorithm-first so the pool fills up
            # with same-algorithm work first.
            future_to_key: dict[Future, tuple] = {
                pool.submit(_run_trial, fmt_name, var, algo): (fmt_name, var, algo)
                for fmt_name, var, algo in work_items
            }

            for fut in as_completed(future_to_key):
                fmt_name, var, algo = future_to_key[fut]
                elapsed = time.perf_counter() - wall_start
                try:
                    rec = fut.result()
                    out_fh.write(json.dumps(rec) + "\n")
                    out_fh.flush()
                    n_records_out += 1
                    w = rec["weights"]
                    print(
                        f"[{n_records_out:3d}/{n_total}]  "
                        f"{fmt_name}/{var}/{algo:<22s}  "
                        f"ratio={w['ratio']:.4f}  "
                        f"c={rec['timing']['compress_time_s']:7.2f}s  "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                except FileNotFoundError as exc:
                    print(f"  SKIP {fmt_name}/{var}/{algo}  [{exc}]", flush=True)
                except Exception as exc:
                    print(f"  ERR  {fmt_name}/{var}/{algo}  [{exc}]", flush=True)
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