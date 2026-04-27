#!/usr/bin/env python3
"""
Lossless compression benchmark for synthetic LLM weight datasets.

Algorithms tested (all at Zstd level 1):
  raw_zstd           Baseline: raw bytes → single Zstd stream
  byte_transpose     One Zstd stream per byte-plane
  semantic_sep       One Zstd stream per semantic field (sign / exp / mantissa)
  bitplane           One Zstd stream per bit-plane (MSB → LSB)
  gorilla_base       Consecutive-element XOR → single Zstd stream

  xor_delta          Base XOR Redraw bytes → single Zstd stream
  gorilla_xor_delta  Gorilla applied to the XOR-delta stream → single Zstd stream

Run policy:
  raw_zstd, byte_transpose, semantic_sep, bitplane, gorilla_base
      → Base split only (Redraw ignored)
  xor_delta, gorilla_xor_delta
      → XOR_Delta split (reads both Base and Redraw)

Multi-stream algorithms (byte_transpose, semantic_sep, bitplane):
  Each portion is compressed independently and sequentially.
  compress_time_s   = time to extract all portions + compress each portion
  decompress_time_s = time to decompress each portion + re-interleave
  compressed_bytes  = sum of all portion compressed sizes

Bit-packing / no-padding rule (semantic_sep, bitplane):
  Each field's bits across ALL elements in a chunk are concatenated into one
  raw bit sequence and then np.packbits'd into bytes — no per-element or
  per-field zero-padding.  Chunk sizes are always a multiple of 8 bits for
  every field (smallest element = 4 bits, chunk = 128 MB = 2^27 bytes,
  so even the 1-bit sign field has 2^27 bits per chunk — byte-aligned).

Parallelism:
  All 42 (format × variance) pairs run simultaneously in a ProcessPoolExecutor
  with MAX_WORKERS = 42.  Each worker owns a unique directory so no two
  workers ever touch the same file.  No individual Zstd call is internally
  parallelised (threads = 0 / default single-threaded).

Output:
  JSONL — one record per (format × variance × split × algorithm).
  weights and scales entries are always present; scales fields are null/0
  for formats without a scales file.
"""

import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zstandard as zstd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path(__file__).parent / "RESULTS_2_30"
OUTPUT_JSONL = Path(__file__).parent / "compression_results.jsonl"

CHUNK_BYTES = 128 * 1024 * 1024   # 128 MB — always byte-aligned for every field
ZSTD_LEVEL  = 1
MAX_WORKERS = 42                   # 7 formats × 6 variances

# ──────────────────────────────────────────────────────────────────────────────
# Dataset structure
# ──────────────────────────────────────────────────────────────────────────────

VARIANCES = ["Var0p01", "Var0p025", "Var0p05", "Var0p1", "Var0p25", "Var0p5"]

# Non-delta algorithms: Base split only
_SINGLE_ALGOS = ["raw_zstd", "byte_transpose", "semantic_sep", "bitplane", "gorilla_base"]
# Delta algorithms: XOR_Delta split (reads Base + Redraw)
_DELTA_ALGOS  = ["xor_delta", "gorilla_xor_delta"]

ALGORITHMS = _SINGLE_ALGOS + _DELTA_ALGOS

# ──────────────────────────────────────────────────────────────────────────────
# Format registry
#
# uint_bits : logical bit-width of one element (4 for FP4, packed 2 per byte)
# n_sign    : number of sign bits (1 for signed floats, 0 for E8M0)
# n_exp     : exponent field width in bits
# n_mant    : mantissa field width in bits
# packed4   : True when two 4-bit elements share one byte
# has_scales: whether a scales.bin companion file exists
# scale_fmt : format descriptor dict for scales.bin (None if no scales)
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
    """Yield successive CHUNK_BYTES raw-byte chunks from a file."""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_BYTES)
            if not chunk:
                break
            yield chunk


def _read_dual_chunks(path_a: Path, path_b: Path):
    """Yield (chunk_a, chunk_b) pairs, stopping at the shorter file."""
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
    """
    Integer dtype for one logical element.
    FP4 (uint_bits=4): no native numpy type; use uint8 (one packed byte = 2 elems).
    """
    byte_width = max(1, fmt["uint_bits"] // 8)
    return np.dtype(f"<u{byte_width}")


# ──────────────────────────────────────────────────────────────────────────────
# Field-extraction helpers
#
# Each returns a list of (field_name, packed_bytes) tuples — one entry per
# independent Zstd stream.  Bit fields are packed with np.packbits with no
# padding: chunk sizes guarantee every field's bit count is a multiple of 8.
# ──────────────────────────────────────────────────────────────────────────────

def _fields_byte_transpose(chunk: bytes, fmt: dict):
    """
    Yield one (name, bytes) stream per byte-plane within each element.
    Yields one plane at a time so the caller can compress-and-discard
    before the next plane is materialised.

    FP32 → 4 streams (byte 0 … byte 3, little-endian)
    FP16/BF16 → 2 streams
    FP8 / packed FP4 → 1 stream (no-op: single-byte elements)
    """
    elem_b = max(1, fmt["uint_bits"] // 8)
    if elem_b == 1:
        yield ("b0", chunk)
        return
    arr = np.frombuffer(chunk, dtype=np.uint8)
    n   = len(arr) // elem_b
    mat = arr[: n * elem_b].reshape(n, elem_b)   # shape (N, elem_b)
    for i in range(elem_b):
        yield (f"b{i}", mat[:, i].tobytes())
        # mat[:, i] is a view; tobytes() copies one plane — previous plane
        # bytes object is released as soon as the caller discards it.


def _fields_semantic_sep(chunk: bytes, fmt: dict):
    """
    Yield one (name, bytes) stream per semantic field: sign / exp / mantissa.
    Yields one plane at a time so the caller can compress-and-discard before
    the next plane is materialised — keeps peak RAM to one plane at a time.

    Bit fields are packed raw (np.packbits) — no padding between elements.
    Chunk byte-alignment guarantees each field's bit count % 8 == 0.

    Yields:
      FP4 packed  → ("sign", ...) then 3x ("mag_bitN", ...)
      E8M0        → ("exp", ...)
      Standard    → ("sign", ...) then n_exp x ("exp_bitN", ...)
                    then n_mant x ("mant_bitN", ...)
    """
    n_sign = fmt["n_sign"]
    n_exp  = fmt["n_exp"]
    n_mant = fmt["n_mant"]

    # ── Packed-nibble FP4 ────────────────────────────────────────────────────
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

    # ── Pure-exponent byte (E8M0) ────────────────────────────────────────────
    if n_sign == 0 and n_mant == 0:
        yield ("exp", chunk)
        return

    # ── Standard float formats ───────────────────────────────────────────────
    dtype = _uint_dtype(fmt)
    arr   = np.frombuffer(chunk, dtype=dtype)   # zero-copy view
    bits  = fmt["uint_bits"]

    # Sign: 1 bit/element
    if n_sign:
        sign_bits = ((arr >> (bits - 1)) & 1).astype(np.uint8)
        yield ("sign", np.packbits(sign_bits).tobytes())
        del sign_bits

    # Exponent: one plane per bit, MSB first
    exp_mask = (1 << n_exp) - 1
    exp_vals = (arr >> n_mant) & exp_mask   # native dtype — no cast needed
    for b in range(n_exp - 1, -1, -1):
        plane = ((exp_vals >> b) & 1).astype(np.uint8)
        yield (f"exp_bit{b}", np.packbits(plane).tobytes())
        del plane
    del exp_vals

    # Mantissa: one plane per bit, MSB first
    if n_mant:
        mant_mask = (1 << n_mant) - 1
        mant_vals = arr & mant_mask          # native dtype — no cast needed
        for b in range(n_mant - 1, -1, -1):
            plane = ((mant_vals >> b) & 1).astype(np.uint8)
            yield (f"mant_bit{b}", np.packbits(plane).tobytes())
            del plane


def _fields_bitplane(chunk: bytes, fmt: dict):
    """
    Yield one (name, bytes) stream per bit-plane of the element integer, MSB→LSB.
    Yields one plane at a time so the caller can compress-and-discard before
    the next plane is materialised.

    For packed FP4: operate on the packed byte (8 planes of the byte word),
    keeping the implementation uniform without nibble unpacking.

    Each plane is np.packbits'd — byte-aligned by the chunk guarantee.
    """
    if fmt["packed4"]:
        arr    = np.frombuffer(chunk, dtype=np.uint8)
        n_bits = 8
    else:
        arr    = np.frombuffer(chunk, dtype=_uint_dtype(fmt))
        n_bits = fmt["uint_bits"]

    for bit_i in range(n_bits - 1, -1, -1):   # MSB first
        plane = ((arr >> bit_i) & 1).astype(np.uint8)
        yield (f"bit{bit_i}", np.packbits(plane).tobytes())
        del plane


# ──────────────────────────────────────────────────────────────────────────────
# Gorilla stateful coder
# ──────────────────────────────────────────────────────────────────────────────

class _Gorilla:
    """Consecutive-element XOR encoder, stateful across chunk boundaries."""

    def __init__(self, fmt: dict):
        self._dt   = _uint_dtype(fmt)
        self._prev = self._dt.type(0)

    def feed(self, data: bytes) -> bytes:
        if not data:
            return b""
        arr = np.frombuffer(data, dtype=self._dt)
        out    = arr.copy()
        out[0] ^= self._prev
        if len(arr) > 1:
            out[1:] ^= arr[:-1]
        self._prev = arr[-1]
        return out.view(np.uint8).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Multi-stream compress / decompress
# ──────────────────────────────────────────────────────────────────────────────

def _compress_multistream(field_chunks_iter_factory, level: int):
    """
    Compress a multi-stream algorithm over a file.

    field_chunks_iter_factory: a zero-argument callable that returns a fresh
      iterable each time it is called.  Each element yielded is itself a
      generator of (name, bytes) tuples — one per stream for that chunk.
      Using a generator (not a list) means only one stream's data is live
      at a time, keeping peak RAM to O(one_plane).

    Returns:
      total_compressed  — sum of compressed sizes across all streams/chunks
      compress_time_s   — field extraction + Zstd compress, full file pass
      decompress_time_s — field extraction + Zstd decompress, full file pass

    Memory contract: at most two objects are live simultaneously:
      • the raw chunk (held by the file reader, released each iteration)
      • one (field_name, field_bytes) tuple from the generator
      • the corresponding compressed blob (created, counted, then del'd)
    No compressed blobs are accumulated across chunks or streams.
    The decompress pass re-reads the file, re-extracts each field, re-compresses
    it (to get the blob to decompress), then immediately decompresses and discards.
    """
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    # ── Compression pass ──────────────────────────────────────────────────────
    total_compressed = 0
    t_compress       = 0.0

    for field_gen in field_chunks_iter_factory():
        t0 = time.perf_counter()
        for _, data in field_gen:           # generator: one plane at a time
            blob = cctx.compress(data)
            total_compressed += len(blob)
            del blob                        # discard immediately
        t_compress += time.perf_counter() - t0

    if total_compressed == 0:
        return 0, 0.0, 0.0

    # ── Decompression pass ────────────────────────────────────────────────────
    # Re-extract each field, re-compress (to get the blob), decompress, discard.
    # Peak memory per iteration: one raw plane + one compressed blob.
    t_decompress = 0.0

    for field_gen in field_chunks_iter_factory():
        t0 = time.perf_counter()
        for _, data in field_gen:
            blob = cctx.compress(data)
            dctx.decompress(blob)
            del blob
        t_decompress += time.perf_counter() - t0

    return total_compressed, t_compress, t_decompress


def _compress_singlestream(chunk_iter, level: int):
    """
    Compress a single-stream algorithm over a file.

    chunk_iter: iterable of preprocessed bytes chunks.

    Returns (compressed_bytes_total, compress_time_s, decompress_time_s).
    """
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    tot_comp = 0
    t_comp   = 0.0
    t_decomp = 0.0

    for chunk in chunk_iter:
        t0 = time.perf_counter()
        compressed = cctx.compress(chunk)
        t_comp += time.perf_counter() - t0

        tot_comp += len(compressed)

        t0 = time.perf_counter()
        dctx.decompress(compressed)
        t_decomp += time.perf_counter() - t0

    return tot_comp, t_comp, t_decomp


# ──────────────────────────────────────────────────────────────────────────────
# Per-algorithm dispatch
# ──────────────────────────────────────────────────────────────────────────────

def _run_algo(algo: str, path: Path, redraw: Path, fmt: dict):
    """
    Run one algorithm on one file.  Returns (comp_bytes, t_comp, t_decomp).

    path   : primary file (Base weights/scales, or Base for delta algos)
    redraw : Redraw file (only used by delta algos)
    fmt    : format descriptor for this data stream
    """
    level = ZSTD_LEVEL

    # ── Multi-stream algorithms ───────────────────────────────────────────────
    if algo == "byte_transpose":
        return _compress_multistream(
            lambda: (_fields_byte_transpose(c, fmt) for c in _read_chunks(path)), level)

    if algo == "semantic_sep":
        return _compress_multistream(
            lambda: (_fields_semantic_sep(c, fmt)   for c in _read_chunks(path)), level)

    if algo == "bitplane":
        return _compress_multistream(
            lambda: (_fields_bitplane(c, fmt)        for c in _read_chunks(path)), level)

    # ── Single-stream algorithms ──────────────────────────────────────────────
    if algo == "raw_zstd":
        return _compress_singlestream(_read_chunks(path), level)

    if algo == "gorilla_base":
        g = _Gorilla(fmt)
        return _compress_singlestream((g.feed(c) for c in _read_chunks(path)), level)

    if algo == "xor_delta":
        xor_iter = (
            (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
            for a, b in _read_dual_chunks(path, redraw)
        )
        return _compress_singlestream(xor_iter, level)

    if algo == "gorilla_xor_delta":
        g = _Gorilla(fmt)
        def _gen():
            for a, b in _read_dual_chunks(path, redraw):
                xb = (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
                yield g.feed(xb)
        return _compress_singlestream(_gen(), level)

    raise ValueError(f"Unknown algorithm: {algo!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Single trial
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(fmt: dict, var: str, algo: str) -> dict:
    """
    Run one (format, variance, algorithm) trial.
    split is always "Base" for non-delta algos, "XOR_Delta" for delta algos.
    Returns a dict ready for JSON serialisation.
    """
    base_dir   = RESULTS_DIR / fmt["name"] / var / "Base"
    redraw_dir = RESULTS_DIR / fmt["name"] / var / "Redraw0p01"

    base_wp   = base_dir   / "weights.bin"
    redraw_wp = redraw_dir / "weights.bin"

    split = "XOR_Delta" if algo in _DELTA_ALGOS else "Base"

    w_orig              = base_wp.stat().st_size
    w_comp, w_tc, w_td  = _run_algo(algo, base_wp, redraw_wp, fmt)

    # ── Scales ────────────────────────────────────────────────────────────────
    s_orig = s_comp = 0
    s_tc   = s_td   = None

    if fmt["has_scales"]:
        base_sp   = base_dir   / "scales.bin"
        redraw_sp = redraw_dir / "scales.bin"
        s_orig             = base_sp.stat().st_size
        s_comp, s_tc, s_td = _run_algo(algo, base_sp, redraw_sp, fmt["scale_fmt"])

    def _ratio(comp, orig):
        return round(comp / orig, 6) if orig > 0 else None

    return dict(
        format     = fmt["name"],
        variance   = var,
        split      = split,
        algorithm  = algo,
        zstd_level = ZSTD_LEVEL,
        weights    = dict(
            original_bytes    = w_orig,
            compressed_bytes  = w_comp,
            ratio             = _ratio(w_comp, w_orig),
            compress_time_s   = w_tc,
            decompress_time_s = w_td,
        ),
        scales     = dict(
            original_bytes    = s_orig,
            compressed_bytes  = s_comp,
            ratio             = _ratio(s_comp, s_orig),
            compress_time_s   = s_tc,
            decompress_time_s = s_td,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# (format × variance) worker — runs in its own subprocess
# ──────────────────────────────────────────────────────────────────────────────

def _run_fmt_var(fmt_name: str, var: str) -> list:
    """
    Run all algorithm trials for one (format, variance) pair sequentially.
    File access is confined to RESULTS/<fmt_name>/<var>/ — unique per worker.
    """
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
                f"c={w['compress_time_s']:7.1f}s  "
                f"d={w['decompress_time_s']:6.1f}s",
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
    work_items = [
        (fmt["name"], var)
        for fmt in FORMATS
        for var in VARIANCES
    ]
    assert len(work_items) == 42

    n_total = len(ALGORITHMS) * len(work_items)

    print(
        f"Output       : {OUTPUT_JSONL}\n"
        f"RESULTS dir  : {RESULTS_DIR}\n"
        f"Chunk size   : {CHUNK_BYTES // 1024 // 1024} MB\n"
        f"Workers      : {MAX_WORKERS}  (7 formats × 6 variances)\n"
        f"Algorithms   : {ALGORITHMS}\n"
        f"Zstd level   : {ZSTD_LEVEL}\n"
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
