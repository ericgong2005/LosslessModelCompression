#!/usr/bin/env python3
"""
Lossless compression benchmark for synthetic LLM weight datasets.

Algorithms tested (each at Zstd levels 1 and 19):
  raw_zstd           Baseline: raw bytes → Zstd
  byte_transpose     Byte-plane transposition → Zstd         (ZipNN core trick)
  semantic_sep       Separate sign/exp/mantissa streams → Zstd
  bitplane           Bit-plane extraction (MSB→LSB) → Zstd
  gorilla_base       Consecutive-element XOR → Zstd           (Gorilla-style)
  xor_delta          Base XOR Redraw bytes → Zstd             (ZipLLM/BitX-style)
  gorilla_xor_delta  Gorilla applied to the XOR-delta stream → Zstd

Parallelism:
  All 42 (format × variance) pairs are dispatched simultaneously to a
  ProcessPoolExecutor with MAX_WORKERS = 42.  Each worker handles one
  (format, variance) pair and runs every (algorithm × level × split) trial
  sequentially within that pair.

  This satisfies the constraint that no two workers ever read the same weight
  file: each (format, variance) directory is unique to exactly one worker.
  No individual Zstd compression call is internally parallelised.

Timing:
  Compression time = preprocessing + Zstd encode (file I/O is inside the
  window but constant across algorithms so comparisons are fair).
  Decompression time = Zstd decode only.

Output:
  JSONL — one record per (format × variance × split × algorithm × level).
  split ∈ {"Base", "Redraw0p01", "XOR_Delta"}
  weights and scales entries are always present; scales fields are null/0
  for formats that have no scales file.
"""

import json
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import zstandard as zstd

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = Path(__file__).parent / "RESULTS_2_10"
OUTPUT_JSONL = Path(__file__).parent / "compression_results.jsonl"

# 128 MB streaming window keeps peak per-worker RAM ≲ ~300 MB even for
# semantic_sep on FP32 (which slightly expands each chunk before Zstd).
CHUNK_BYTES = 128 * 1024 * 1024

# 6 variances × 7 formats = 42 independent (format, variance) pairs.
# Each pair maps to a unique directory so workers never share a file.
MAX_WORKERS = 42

# ──────────────────────────────────────────────────────────────────────────────
# Dataset structure
# ──────────────────────────────────────────────────────────────────────────────

VARIANCES = ["Var0p01", "Var0p025", "Var0p05", "Var0p1", "Var0p25", "Var0p5"]

ZSTD_LEVELS = [1, 19]

ALGORITHMS = [
    "raw_zstd",
    "byte_transpose",
    "semantic_sep",
    "bitplane",
    "gorilla_base",
    "xor_delta",
    "gorilla_xor_delta",
]

# Algorithms that consume both Base and Redraw files
_DELTA_ALGOS = {"xor_delta", "gorilla_xor_delta"}

# ──────────────────────────────────────────────────────────────────────────────
# Format registry
#
# uint_bits : logical bit-width of one element (4 for FP4 — packed 2/byte)
# n_sign    : 1 for signed floats, 0 for pure-exponent formats (E8M0)
# n_exp     : exponent field width in bits
# n_mant    : mantissa field width in bits
# packed4   : True when two 4-bit elements share one byte
# has_scales: whether a scales.bin companion file exists
# scale_fmt : descriptor dict for scales.bin  (None if no scales)
# ──────────────────────────────────────────────────────────────────────────────

_SFMT_FP32    = dict(uint_bits=32, n_sign=1, n_exp=8, n_mant=23, packed4=False)
_SFMT_E8M0    = dict(uint_bits=8,  n_sign=0, n_exp=8, n_mant=0,  packed4=False)
_SFMT_FP8E4M3 = dict(uint_bits=8,  n_sign=1, n_exp=4, n_mant=3,  packed4=False)

FORMATS = [
    dict(name="FP32E8M23",  uint_bits=32, n_sign=1, n_exp=8, n_mant=23,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="FP16E5M10",  uint_bits=16, n_sign=1, n_exp=5, n_mant=10,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="BF16E8M7",   uint_bits=16, n_sign=1, n_exp=8, n_mant=7,
         packed4=False, has_scales=False, scale_fmt=None),
    dict(name="FP8E4M3",    uint_bits=8,  n_sign=1, n_exp=4, n_mant=3,
         packed4=False, has_scales=True,  scale_fmt=_SFMT_FP32),
    dict(name="FP8E5M2",    uint_bits=8,  n_sign=1, n_exp=5, n_mant=2,
         packed4=False, has_scales=True,  scale_fmt=_SFMT_FP32),
    dict(name="MXFP4E2M1",  uint_bits=4,  n_sign=1, n_exp=2, n_mant=1,
         packed4=True,  has_scales=True,  scale_fmt=_SFMT_E8M0),
    dict(name="NVFP4E2M1",  uint_bits=4,  n_sign=1, n_exp=2, n_mant=1,
         packed4=True,  has_scales=True,  scale_fmt=_SFMT_FP8E4M3),
]

# Lookup dict used inside worker subprocesses to retrieve format info by name
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
    """
    Yield (chunk_a, chunk_b) pairs from two files, stopping at the shorter.
    Used by delta algorithms that must read Base and Redraw simultaneously.
    """
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
    FP4 formats (uint_bits=4) have no native numpy type; we use uint8 and
    treat each packed byte (two nibbles) as the atomic unit.
    """
    byte_width = max(1, fmt["uint_bits"] // 8)   # 4 // 8 = 0 → clamp to 1
    return np.dtype(f"<u{byte_width}")


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing functions
# Each takes raw bytes + a format descriptor, returns preprocessed bytes.
# ──────────────────────────────────────────────────────────────────────────────

def _pp_raw(data: bytes, _fmt: dict) -> bytes:
    """No-op: passes raw bytes straight to the compressor."""
    return data


def _pp_byte_transpose(data: bytes, fmt: dict) -> bytes:
    """
    Regroup bytes by their position within each element (ZipNN core trick).

    FP32 example (4 bytes/elem, little-endian):
      Original  : [b0 b1 b2 b3 | b0 b1 b2 b3 | ...]
      Transposed : [b0 b0 ... | b1 b1 ... | b2 b2 ... | b3 b3 ...]

    Byte 3 of little-endian FP32/BF16 holds the sign bit + all 8 exponent
    bits.  After transposition it forms one long, highly-compressible block.

    No-op for 1-byte-per-element formats (FP8, packed FP4).
    """
    elem_b = max(1, fmt["uint_bits"] // 8)
    if elem_b == 1:
        return data
    arr = np.frombuffer(data, dtype=np.uint8)
    n   = len(arr) // elem_b
    if n == 0:
        return data
    return arr[: n * elem_b].reshape(n, elem_b).T.flatten().tobytes()


def _pp_semantic_sep(data: bytes, fmt: dict) -> bytes:
    """
    Separate the IEEE 754 bit-fields (sign / exponent / mantissa) into
    independent streams, then concatenate for a single Zstd pass.

    Output layout:
      packed sign bits   ceil(N/8) bytes      — 1 bit/element via np.packbits
      exponent bytes     N bytes              — uint8 (n_exp ≤ 8 always)
      mantissa values    N × ceil(n_mant/8) B — uint8, uint16, or trimmed uint32

    Special cases:
      FP4 packed : sign bits packed + 3-bit magnitude-index bytes per nibble
      E8M0 scale : whole byte IS the exponent → returned unchanged (no-op)
    """
    n_sign = fmt["n_sign"]
    n_exp  = fmt["n_exp"]
    n_mant = fmt["n_mant"]

    # ── Packed-nibble FP4 formats ────────────────────────────────────────────
    if fmt["packed4"]:
        packed  = np.frombuffer(data, dtype=np.uint8)
        lo      = packed & 0x0F
        hi      = (packed >> 4) & 0x0F
        nibbles = np.empty(len(packed) * 2, dtype=np.uint8)
        nibbles[0::2], nibbles[1::2] = lo, hi
        # E2M1 nibble: bit 3 = sign, bits 2:0 = magnitude-codebook index
        signs   = np.packbits((nibbles >> 3) & 1)
        indices = (nibbles & 0x7).astype(np.uint8)
        return signs.tobytes() + indices.tobytes()

    # ── Pure-exponent byte (E8M0 scale format) ───────────────────────────────
    if n_sign == 0 and n_mant == 0:
        return data   # whole byte is already the exponent

    # ── Standard float formats ───────────────────────────────────────────────
    dtype = _uint_dtype(fmt)
    arr   = np.frombuffer(data, dtype=dtype)
    bits  = fmt["uint_bits"]
    out   = b""

    # Sign: 1 bit per element → packed into ceil(N/8) bytes
    if n_sign:
        sign_bits = ((arr >> (bits - 1)) & 1).astype(np.uint8)
        out += np.packbits(sign_bits).tobytes()

    # Exponent: n_exp ≤ 8 bits → fits in uint8
    exp_mask = (1 << n_exp) - 1
    out += ((arr >> n_mant) & exp_mask).astype(np.uint8).tobytes()

    # Mantissa: minimal byte representation
    if n_mant:
        mant = arr & ((1 << n_mant) - 1)
        if n_mant <= 8:
            out += mant.astype(np.uint8).tobytes()
        elif n_mant <= 16:
            out += mant.astype(np.uint16).tobytes()
        else:
            # n_mant = 23 (FP32): store in 3 bytes by dropping the always-zero
            # MSByte of uint32.  Little-endian layout: bytes [0:3] hold all 23 bits.
            u32 = mant.astype(np.uint32)
            out += u32.view(np.uint8).reshape(-1, 4)[:, :3].flatten().tobytes()

    return out


def _pp_bitplane(data: bytes, fmt: dict) -> bytes:
    """
    Collect all bit-N values together, MSB down to LSB.

    Each plane is bit-packed via np.packbits → ceil(N/8) bytes per plane.
    High-order planes (sign + exponent bits) compress well; low-order
    mantissa planes are near-incompressible.

    For packed FP4: operates on the packed byte as 8 planes rather than
    unpacking to 4 nibble planes, keeping the implementation uniform.
    """
    if fmt["packed4"]:
        arr    = np.frombuffer(data, dtype=np.uint8)
        n_bits = 8
    else:
        arr    = np.frombuffer(data, dtype=_uint_dtype(fmt))
        n_bits = fmt["uint_bits"]

    planes = bytearray()
    for bit_i in range(n_bits - 1, -1, -1):   # MSB first
        plane = ((arr >> bit_i) & 1).astype(np.uint8)
        planes.extend(np.packbits(plane).tobytes())
    return bytes(planes)


# ──────────────────────────────────────────────────────────────────────────────
# Gorilla-style consecutive-element XOR coder  (stateful across chunks)
# ──────────────────────────────────────────────────────────────────────────────

class _Gorilla:
    """
    Encodes a stream of integers as consecutive XOR differences.

    Element 0 of the full file is stored as-is (XOR'd against 0).
    Every subsequent element is XOR'd against its predecessor.
    State (_prev) is carried across chunk boundaries for continuity.

    Adjacent weights within a layer often differ only in low mantissa bits,
    so the XOR stream is sparse and compresses well with Zstd.
    """

    def __init__(self, fmt: dict):
        self._dt   = _uint_dtype(fmt)
        self._prev = self._dt.type(0)

    def feed(self, data: bytes) -> bytes:
        if not data:
            return b""
        arr = np.frombuffer(data, dtype=self._dt)
        if len(arr) == 0:
            return b""
        out    = arr.copy()          # mutable copy; arr remains read-only
        out[0] ^= self._prev
        if len(arr) > 1:
            out[1:] ^= arr[:-1]     # reads original arr → no aliasing issue
        self._prev = arr[-1]
        return out.view(np.uint8).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Per-algorithm chunk-iterator factories
# ──────────────────────────────────────────────────────────────────────────────

def _gorilla_xor_delta_gen(path_a: Path, path_b: Path, fmt: dict):
    """
    Generator: Gorilla-encode the element-wise XOR-delta stream.

    Step 1 — XOR delta:
      D[i] = base[i] XOR redraw[i]  (byte-level XOR == element-level XOR)
      For 1% fine-tuning: ~99% of D[i] are zero.

    Step 2 — Gorilla:
      G[i] = D[i] XOR D[i-1]
            = (base[i] XOR redraw[i]) XOR (base[i-1] XOR redraw[i-1])

      G is nonzero only at change-boundary transitions, making it sparser
      still than the raw delta and extremely Zstd-friendly.
    """
    g = _Gorilla(fmt)
    for ca, cb in _read_dual_chunks(path_a, path_b):
        xor_bytes = (
            np.frombuffer(ca, np.uint8) ^ np.frombuffer(cb, np.uint8)
        ).tobytes()
        yield g.feed(xor_bytes)


def _make_iter(algo: str, primary: Path, redraw: Path, fmt: dict):
    """
    Return a lazy chunk iterator for the given algorithm.

    primary : file to read for this split (Base or Redraw weights/scales)
    redraw  : Redraw file (used only by delta algorithms)
    fmt     : format descriptor for this data stream
    """
    if algo == "raw_zstd":
        return (_pp_raw(c, fmt)            for c in _read_chunks(primary))

    if algo == "byte_transpose":
        return (_pp_byte_transpose(c, fmt) for c in _read_chunks(primary))

    if algo == "semantic_sep":
        return (_pp_semantic_sep(c, fmt)   for c in _read_chunks(primary))

    if algo == "bitplane":
        return (_pp_bitplane(c, fmt)       for c in _read_chunks(primary))

    if algo == "gorilla_base":
        g = _Gorilla(fmt)
        return (g.feed(c)                  for c in _read_chunks(primary))

    if algo == "xor_delta":
        return (
            (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()
            for a, b in _read_dual_chunks(primary, redraw)
        )

    if algo == "gorilla_xor_delta":
        return _gorilla_xor_delta_gen(primary, redraw, fmt)

    raise ValueError(f"Unknown algorithm: {algo!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Core timing loop
# ──────────────────────────────────────────────────────────────────────────────

def _compress_decompress(chunk_iter, level: int):
    """
    Consume a preprocessed chunk iterator.  Per chunk:
      1. Compress with Zstd (timed).
      2. Immediately decompress (timed separately).
      3. Discard both — at most one compressed chunk lives in memory at once.

    Returns (compressed_bytes_total, compress_s, decompress_s).
    """
    cctx = zstd.ZstdCompressor(level=level)   # single-threaded by default
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
# Single-trial runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_trial(fmt: dict, var: str, split: str, algo: str, level: int) -> dict:
    """
    Run one (format, variance, split, algorithm, zstd_level) trial.
    Returns a dict ready for JSON serialisation.
    """
    base_dir   = RESULTS_DIR / fmt["name"] / var / "Base"
    redraw_dir = RESULTS_DIR / fmt["name"] / var / "Redraw0p01"

    base_wp   = base_dir   / "weights.bin"
    redraw_wp = redraw_dir / "weights.bin"

    # primary_wp is used for original_bytes accounting.
    # Delta algorithms read both files; base file size is used for them.
    primary_wp = redraw_wp if split == "Redraw0p01" else base_wp

    w_orig              = primary_wp.stat().st_size
    w_iter              = _make_iter(algo, primary_wp, redraw_wp, fmt)
    w_comp, w_tc, w_td  = _compress_decompress(w_iter, level)

    # ── Scales (if this format has them) ─────────────────────────────────────
    s_orig = s_comp = 0
    s_tc   = s_td   = None

    if fmt["has_scales"]:
        base_sp    = base_dir   / "scales.bin"
        redraw_sp  = redraw_dir / "scales.bin"
        primary_sp = redraw_sp if split == "Redraw0p01" else base_sp

        s_orig              = primary_sp.stat().st_size
        s_iter              = _make_iter(algo, primary_sp, redraw_sp, fmt["scale_fmt"])
        s_comp, s_tc, s_td  = _compress_decompress(s_iter, level)

    def _ratio(comp, orig):
        return round(comp / orig, 6) if orig > 0 else None

    return dict(
        format     = fmt["name"],
        variance   = var,
        split      = split,
        algorithm  = algo,
        zstd_level = level,
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
            compress_time_s   = s_tc if s_tc is not None else None,
            decompress_time_s = s_td if s_td is not None else None,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# (format × variance) worker — executes in its own subprocess
# ──────────────────────────────────────────────────────────────────────────────

def _run_fmt_var(fmt_name: str, var: str) -> list:
    """
    Worker entry point.  Runs all (algorithm × level × split) trials for one
    (format, variance) pair sequentially, then returns the collected records.

    All file access is confined to:
      RESULTS/<fmt_name>/<var>/Base/
      RESULTS/<fmt_name>/<var>/Redraw0p01/
    which are unique to this worker, guaranteeing no cross-worker file sharing.
    """
    fmt     = _FMT_BY_NAME[fmt_name]
    records = []

    for algo in ALGORITHMS:
        splits = ["XOR_Delta"] if algo in _DELTA_ALGOS else ["Base", "Redraw0p01"]
        for level in ZSTD_LEVELS:
            for split in splits:
                label = f"{fmt_name}/{var}/{split}/{algo}/L{level}"
                try:
                    rec = _run_trial(fmt, var, split, algo, level)
                    records.append(rec)
                    w = rec["weights"]
                    print(
                        f"  OK   {label:<60s}  "
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
    # Build the full list of 42 (format × variance) work items
    work_items = [
        (fmt["name"], var)
        for fmt in FORMATS
        for var in VARIANCES
    ]
    assert len(work_items) == len(FORMATS) * len(VARIANCES), "Expected 42 work items"

    # Count total trials for progress display
    trials_per_pair = sum(
        len(["XOR_Delta"] if a in _DELTA_ALGOS else ["Base", "Redraw0p01"])
        * len(ZSTD_LEVELS)
        for a in ALGORITHMS
    )
    n_total = trials_per_pair * len(work_items)

    print(
        f"Output       : {OUTPUT_JSONL}\n"
        f"RESULTS dir  : {RESULTS_DIR}\n"
        f"Chunk size   : {CHUNK_BYTES // 1024 // 1024} MB\n"
        f"Workers      : {MAX_WORKERS}  "
        f"({len(FORMATS)} formats × {len(VARIANCES)} variances)\n"
        f"Formats      : {[f['name'] for f in FORMATS]}\n"
        f"Variances    : {VARIANCES}\n"
        f"Algorithms   : {ALGORITHMS}\n"
        f"Zstd levels  : {ZSTD_LEVELS}\n"
        f"Trials/pair  : {trials_per_pair}\n"
        f"Total trials : {n_total}\n",
        flush=True,
    )

    wall_start    = time.perf_counter()
    n_records_out = 0
    n_pairs_done  = 0

    with open(OUTPUT_JSONL, "w") as out_fh:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # Submit all 42 workers at once
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