#!/usr/bin/env python3
"""
Unified plotting script — takes both a Huffman and a Zstd JSONL results file
and produces plots under PLOTS_UNIFIED/.

Output structure
────────────────
PLOTS_UNIFIED/
  VarianceTable.Rmd
  Overall/
    algo_bars_ratio_base.png
    algo_bars_ratio_delta.png
    algo_bars_compress_time_base.png
    algo_bars_compress_time_delta.png
    algo_bars_decompress_time_base.png
    algo_bars_decompress_time_delta.png
  semantic_sep/
    ratio.png  compress_time.png  decompress_time.png
      x-axis : w_sign, w_exp, w_mant, s_sign, s_exp, s_mant
      bars   : one per quantization (7 bars per tick, omit scale ticks for
               formats without scales)
  byte_transpose_{FMT}/          (one folder per quantization × codec pair)
    ratio.png  compress_time.png  decompress_time.png
      x-axis : w_b0, w_b1, … s_b0, s_b1, … (scale section omitted if none)
      bars   : one bar per stream  (error bars = std over variances)
  bitplane_{FMT}/
    (same structure as byte_transpose)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# ① INPUT FILES
# ──────────────────────────────────────────────────────────────────────────────

HUFFMAN_JSONL = "compression_results_huffman.jsonl"
ZSTD_JSONL    = "compression_results_zstd.jsonl"

OUTPUT_DIR = Path("PLOTS_UNIFIED")

# ──────────────────────────────────────────────────────────────────────────────
# ② DISPLAY LABELS
# ──────────────────────────────────────────────────────────────────────────────

LABELS = {
    "formats": {
        "FP32E8M23":  "FP32:E8M23",
        "BF16E8M7":   "BF16:E8M7",
        "FP16E5M10":  "FP16:E5M10",
        "FP8E5M2":    "FP8:E5M2",
        "FP8E4M3":    "FP8:E4M3",
        "MXFP4E2M1":  "MXFP4:E2M1",
        "NVFP4E2M1":  "NVFP4:E2M1",
    },
    "algorithms": {
        "raw_zstd":          "Baseline",
        "raw_huffman":       "Baseline",
        "byte_transpose":    "Byte-split",
        "semantic_sep":      "Semantic-split",
        "bitplane":          "Bit-split",
        "gorilla_base":      "Gorilla",
        "xor_delta":         "XOR Delta",
        "gorilla_xor_delta": "Gorilla XOR Delta",
    },
    "codecs": {
        "huffman": "Huffman",
        "zstd":    "Zstd",
    },
    "metrics": {
        "weights_ratio":           "Compression Ratio",
        "total_compress_time_s":   "Compression Throughput (GB/s)",
        "total_decompress_time_s": "Decompression Throughput (GB/s)",
    },
    "x_axes": {
        "algorithm":  "",
        "variance":   "Weight Variance (\u03c3\u00b2)",
        "stream":     "Stream",
    },
    "legends": {
        "format":     "Quantization",
        "algo_codec": "Algorithm",
        "codec":      "Compressor",
    },
    "bar_titles": {
        ("ratio",          "base"):  "{metric}\nNon-delta Algorithms",
        ("ratio",          "delta"): "{metric}\nDelta Algorithms",
        ("compress_time",  "base"):  "{metric}\nNon-delta Algorithms",
        ("compress_time",  "delta"): "{metric}\nDelta Algorithms",
        ("decompress_time","base"):  "{metric}\nNon-delta Algorithms",
        ("decompress_time","delta"): "{metric}\nDelta Algorithms",
    },
    "bar_subtitle":    "\u00b11 std across variances",
    "scatter_titles": {
        "ratio":          "{metric} vs. Weight Variance",
        "compress_time":  "{metric} vs. Weight Variance",
        "decompress_time":"{metric} vs. Weight Variance",
    },
    "rmd_title":      "Variance Independence Tables",
    "rmd_row_header": "Algorithm / Compressor",
    "rmd_caption": (
        "Each cell shows $\\text{mean} \\pm \\text{std}$ "
        "across the 6 weight variance levels "
        "(averaged over splits before computing statistics)."
    ),
}


def _fmt_label(fmt: str) -> str:
    return LABELS["formats"].get(fmt, fmt)

def _algo_label(algo: str, codec: str) -> str:
    a = LABELS["algorithms"].get(algo, algo)
    c = LABELS["codecs"].get(codec, codec)
    return f"{a}\n{c}"

# ──────────────────────────────────────────────────────────────────────────────
# ③ VISUAL IDENTITY
# ──────────────────────────────────────────────────────────────────────────────

FORMAT_ORDER = [
    "FP32E8M23", "BF16E8M7", "FP16E5M10",
    "FP8E5M2", "FP8E4M3", "MXFP4E2M1", "NVFP4E2M1",
]

FORMAT_ORDER_ASC = list(reversed(FORMAT_ORDER))   # low → high precision

# Formats that have a scales.bin companion
SCALED_FORMATS = {"FP8E4M3", "FP8E5M2", "MXFP4E2M1", "NVFP4E2M1"}

ALGORITHM_ORDER = [
    "raw_zstd", "raw_huffman", "byte_transpose",
    "semantic_sep", "bitplane", "gorilla_base",
    "xor_delta", "gorilla_xor_delta",
]

_PALETTE_14 = [
    "#e6194b", "#f58231", "#3cb44b", "#008000",
    "#4363d8", "#42d4f4", "#911eb4", "#dcbeff",
    "#9a6324", "#f032e6", "#800000", "#aaffc3",
    "#469990", "#000075",
]
_MARKERS_7 = ["o", "s", "^", "D", "v", "P", "X"]

# Muted gradient for quantization bars (low → high precision = blue → sienna)
_FMT_COLORS = [
    "#4e79a7",  # NVFP4E2M1
    "#6ba3be",  # MXFP4E2M1
    "#86bcb6",  # FP8E4M3
    "#b0c4a0",  # FP8E5M2
    "#d4a96a",  # FP16E5M10
    "#c47d3e",  # BF16E8M7
    "#8c4a2f",  # FP32E8M23
]
_FMT_COLOR_MAP = dict(zip(FORMAT_ORDER_ASC, _FMT_COLORS))

# Two muted colours for the two codecs in per-stream plots
_CODEC_COLORS = {"huffman": "#5a7fc4", "zstd": "#c47d3e"}

_BASE_ALGOS      = {"raw_zstd","raw_huffman","byte_transpose","semantic_sep","bitplane","gorilla_base"}
_DELTA_ALGOS_SET = {"xor_delta","gorilla_xor_delta"}

# ──────────────────────────────────────────────────────────────────────────────
# ④ LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl_flat(path: str, codec: str) -> pd.DataFrame:
    """Flat per-record DataFrame (aggregate weights/scales only)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {ln}: {e}") from e
            w = obj.get("weights", {}) or {}
            s = obj.get("scales",  {}) or {}
            t = obj.get("timing",  {}) or {}
            rows.append({
                "codec":     codec,
                "format":    obj.get("format"),
                "variance":  obj.get("variance"),
                "split":     obj.get("split"),
                "algorithm": obj.get("algorithm"),
                "weights_ratio":             w.get("ratio"),
                "weights_compress_time_s":   w.get("compress_time_s"),
                "weights_decompress_time_s": w.get("decompress_time_s"),
                "weights_original_bytes":    w.get("original_bytes"),
                "scales_ratio":              s.get("ratio"),
                "scales_compress_time_s":    s.get("compress_time_s"),
                "scales_decompress_time_s":  s.get("decompress_time_s"),
                "scales_original_bytes":     s.get("original_bytes"),
                "timing_compress_time_s":    t.get("compress_time_s"),
                "timing_decompress_time_s":  t.get("decompress_time_s"),
            })
    if not rows:
        raise ValueError(f"No rows in {path}")
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("codec","format","variance","split","algorithm"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_jsonl_streams(path: str, codec: str) -> pd.DataFrame:
    """
    Explode the per-stream data inside weights.streams and scales.streams
    into one row per (record × data_key × stream_name).

    Columns: codec, format, variance, split, algorithm,
             data_key ("weights"|"scales"), stream, ratio,
             compress_time_s, decompress_time_s
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            base = {
                "codec":     codec,
                "format":    obj.get("format"),
                "variance":  obj.get("variance"),
                "split":     obj.get("split"),
                "algorithm": obj.get("algorithm"),
            }
            for dk in ("weights", "scales"):
                block   = obj.get(dk, {}) or {}
                streams = block.get("streams", {}) or {}
                for sname, sv in streams.items():
                    if not isinstance(sv, dict):
                        continue
                    ob  = sv.get("original_bytes")
                    ctm = sv.get("compress_time_s")
                    dtm = sv.get("decompress_time_s")
                    rows.append({
                        **base,
                        "data_key":                   dk,
                        "stream":                     sname,
                        "ratio":                      sv.get("ratio"),
                        "compress_time_s":            ctm,
                        "decompress_time_s":          dtm,
                        "original_bytes":             ob,
                        "compressed_bytes":           sv.get("compressed_bytes"),
                        "compress_throughput_gb_s":   _throughput_gb_s(ob, ctm),
                        "decompress_throughput_gb_s": _throughput_gb_s(ob, dtm),
                    })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("ratio","compress_time_s","decompress_time_s",
              "original_bytes","compressed_bytes",
              "compress_throughput_gb_s","decompress_throughput_gb_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_both():
    flat_dfs   = []
    stream_dfs = []
    for path, codec in [(HUFFMAN_JSONL,"huffman"), (ZSTD_JSONL,"zstd")]:
        try:
            flat_dfs.append(load_jsonl_flat(path, codec))
            stream_dfs.append(load_jsonl_streams(path, codec))
            print(f"  Loaded {codec}: {path}")
        except FileNotFoundError:
            print(f"  WARNING: {path} not found — skipping {codec}")
    if not flat_dfs:
        raise RuntimeError("No input files found.")
    flat   = pd.concat(flat_dfs,   ignore_index=True)
    streams = pd.concat(stream_dfs, ignore_index=True) if stream_dfs else pd.DataFrame()
    return flat, streams

# ──────────────────────────────────────────────────────────────────────────────
# ⑤ HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def variance_to_float(s: str) -> float:
    try:
        return float(s.replace("Var","").replace("var","").replace("p","."))
    except Exception:
        return float("nan")


def _total_time(row, col_prefix: str) -> float:
    top = row.get(f"timing_{col_prefix}_time_s")
    if pd.notna(top):
        return float(top)
    w = row.get(f"weights_{col_prefix}_time_s")
    s = row.get(f"scales_{col_prefix}_time_s")
    return (float(w) if pd.notna(w) else 0.0) + (float(s) if pd.notna(s) else 0.0)


def _throughput_gb_s(original_bytes, time_s) -> float:
    """Compute throughput in GB/s (1 GB = 1e9 bytes). Returns NaN if invalid."""
    try:
        b = float(original_bytes)
        t = float(time_s)
        if t > 0 and b > 0:
            return (b / 1e9) / t
        return float("nan")
    except (TypeError, ValueError):
        return float("nan")


def build_color_map(df: pd.DataFrame):
    combos, codecs_present = [], sorted(df["codec"].dropna().unique())
    for algo in ALGORITHM_ORDER:
        for codec in codecs_present:
            if ((df["algorithm"]==algo)&(df["codec"]==codec)).any():
                combos.append((algo,codec))
    return {k: _PALETTE_14[i] if i<len(_PALETTE_14) else "#888888"
            for i,k in enumerate(combos)}


def build_marker_map(df: pd.DataFrame):
    fmts    = [f for f in FORMAT_ORDER if f in df["format"].unique()]
    extras  = [f for f in df["format"].dropna().unique() if f not in fmts]
    ordered = fmts + extras
    return {fmt: _MARKERS_7[i%len(_MARKERS_7)] for i,fmt in enumerate(ordered)}


def _stream_sort_key(name: str) -> tuple:
    """Logical sort order for stream names.
    magnitude_index (FP4 E2M1) encodes both exponent and mantissa jointly,
    so it is grouped with the exponent field, after exp but before mant.
    """
    if name == "sign":            return (0,)
    if name == "exp":             return (1,)
    if name == "magnitude_index": return (1,)   # treated as exponent
    if name == "mant":            return (2,)
    if name.startswith("exp_bit"):
        try: return (1, -int(name[7:]))
        except: pass
    if name.startswith("mant_bit"):
        try: return (2, -int(name[8:]))
        except: pass
    if name.startswith("mag_bit"):
        try: return (1, 9000 - int(name[7:]))   # after exp bits, before mant
        except: pass
    if name.startswith("b") and name[1:].isdigit():
        return (10, int(name[1:]))
    if name.startswith("bit") and name[3:].isdigit():
        return (20, -int(name[3:]))
    if name == "data":            return (99,)
    return (50, name)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ VARIANCE TABLE
# ──────────────────────────────────────────────────────────────────────────────

def make_variance_tables(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["total_compress_time_s"]   = df.apply(lambda r: _total_time(r,"compress"),   axis=1)
    df["total_decompress_time_s"] = df.apply(lambda r: _total_time(r,"decompress"), axis=1)
    df["total_original_bytes"] = (
        df["weights_original_bytes"].fillna(0) +
        df["scales_original_bytes"].fillna(0)
    )
    df["compress_throughput_gb_s"]   = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_compress_time_s"]),   axis=1)
    df["decompress_throughput_gb_s"] = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_decompress_time_s"]), axis=1)

    metrics = [
        ("weights_ratio",             LABELS["metrics"]["weights_ratio"]),
        ("compress_throughput_gb_s",  LABELS["metrics"]["total_compress_time_s"]),
        ("decompress_throughput_gb_s",LABELS["metrics"]["total_decompress_time_s"]),
    ]
    fmts_present = [f for f in FORMAT_ORDER if f in df["format"].unique()]
    all_fmts     = fmts_present + sorted(
        f for f in df["format"].dropna().unique() if f not in FORMAT_ORDER)
    row_keys = [(algo,codec)
                for algo in ALGORITHM_ORDER
                for codec in sorted(df["codec"].dropna().unique())
                if ((df["algorithm"]==algo)&(df["codec"]==codec)).any()]

    lines = ["---", f'title: "{LABELS["rmd_title"]}"', "output: html_document", "---", ""]

    for metric_col, metric_label in metrics:
        lines += [f"## {metric_label}", "", LABELS["rmd_caption"], ""]
        _df_m = df if metric_col != "decompress_throughput_gb_s" else df[df["codec"] != "huffman"]
        cell_means = (_df_m.dropna(subset=[metric_col])
                      .groupby(["algorithm","codec","format","variance"], observed=True)
                      [metric_col].mean().reset_index())
        agg = (cell_means.groupby(["algorithm","codec","format"], observed=True)
               [metric_col].agg(["mean","std"]))
        agg.columns = ["mean","std"]

        lines.append("| " + LABELS["rmd_row_header"] + " | " +
                     " | ".join(_fmt_label(f) for f in all_fmts) + " |")
        lines.append("| :--- | " + " | ".join([":---:"]*len(all_fmts)) + " |")

        for algo, codec in row_keys:
            rl = (f"{LABELS['algorithms'].get(algo,algo)}\n"
                  f"{LABELS['codecs'].get(codec,codec)}")
            cells = []
            for fmt in all_fmts:
                try:
                    r = agg.loc[(algo,codec,fmt)]
                    mu, sd = r["mean"], r["std"]
                    if pd.notna(mu) and pd.notna(sd):
                        cells.append(f"$\\small {mu:.4f} \\pm {sd:.4f}$")
                    elif pd.notna(mu):
                        cells.append(f"$\\small {mu:.4f}$")
                    else:
                        cells.append("—")
                except KeyError:
                    cells.append("—")
            lines.append(f"| {rl} | " + " | ".join(cells) + " |")
        lines.append("")

    out_path = out_dir / "VarianceTable.Rmd"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ OVERALL BAR PLOTS  (saved under Overall/)
# ──────────────────────────────────────────────────────────────────────────────

def plot_algorithm_bars(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["total_compress_time_s"]   = df.apply(lambda r: _total_time(r,"compress"),   axis=1)
    df["total_decompress_time_s"] = df.apply(lambda r: _total_time(r,"decompress"), axis=1)
    # total original bytes = weights + scales (where present)
    df["total_original_bytes"] = (
        df["weights_original_bytes"].fillna(0) +
        df["scales_original_bytes"].fillna(0)
    )
    df["compress_throughput_gb_s"]   = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_compress_time_s"]),   axis=1)
    df["decompress_throughput_gb_s"] = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_decompress_time_s"]), axis=1)

    metrics = [
        ("weights_ratio",             LABELS["metrics"]["weights_ratio"],             "ratio"),
        ("compress_throughput_gb_s",  LABELS["metrics"]["total_compress_time_s"],     "compress_time"),
        ("decompress_throughput_gb_s",LABELS["metrics"]["total_decompress_time_s"],   "decompress_time"),
    ]
    groups  = [("base",_BASE_ALGOS,"Non-delta"),("delta",_DELTA_ALGOS_SET,"Delta")]
    fmts    = [f for f in FORMAT_ORDER_ASC if f in df["format"].unique()]

    for metric_col, metric_label, metric_tag in metrics:
        for group_tag, algo_set, group_title in groups:
            _df = df if metric_tag != "decompress_time" else df[df["codec"] != "huffman"]
            cell = (_df[_df["algorithm"].isin(algo_set)]
                    .dropna(subset=[metric_col])
                    .groupby(["algorithm","codec","format","variance"], observed=True)
                    [metric_col].mean().reset_index())
            if cell.empty:
                continue
            agg = (cell.groupby(["algorithm","codec","format"], observed=True)
                   [metric_col].agg(["mean","std"]).reset_index())
            agg.columns = ["algorithm","codec","format","mean","std"]

            algo_mean = (agg.groupby(["algorithm","codec"], observed=True)["mean"]
                         .mean().reset_index().sort_values("mean"))
            algo_keys   = [(r["algorithm"],r["codec"]) for _,r in algo_mean.iterrows()]
            algo_labels = [_algo_label(a,c) for a,c in algo_keys]

            n_algos = len(algo_keys)
            n_fmts  = len(fmts)
            bar_w   = 0.82 / n_fmts
            x       = np.arange(n_algos)

            fig, ax = plt.subplots(figsize=(max(10, n_algos*1.4+3), 6))
            for fi, fmt in enumerate(fmts):
                offsets = x - 0.41 + (fi+0.5)*bar_w
                means = []; stds = []
                for algo, codec in algo_keys:
                    row = agg[(agg["algorithm"]==algo)&(agg["codec"]==codec)&(agg["format"]==fmt)]
                    means.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
                    stds.append(float(row["std"].fillna(0).iloc[0]) if not row.empty else 0.0)
                ax.bar(offsets, means, width=bar_w,
                       color=_FMT_COLOR_MAP.get(fmt,"#888"), label=_fmt_label(fmt), alpha=0.88)
                ax.errorbar(offsets, means, yerr=stds, fmt="none",
                            ecolor="black", elinewidth=0.8, capsize=2, zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(algo_labels, fontsize=8, rotation=0, ha="center")
            ax.set_xlabel(LABELS["x_axes"]["algorithm"], fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            tmpl = LABELS["bar_titles"].get((metric_tag,group_tag),"{metric}\n{group}")
            ax.legend(title=LABELS["legends"]["format"],
                      bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8, title_fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
            if metric_tag in ("compress_time", "decompress_time"):
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            fig.tight_layout()
            _save(fig, out_dir / f"algo_bars_{metric_tag}_{group_tag}.png")


# ──────────────────────────────────────────────────────────────────────────────
# ⑧ PER-STREAM HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_STREAM_METRICS = [
    ("ratio",                       "Compression Ratio",            "ratio"),
    ("compress_throughput_gb_s",    "Compression Throughput (GB/s)","compress_time"),
    ("decompress_throughput_gb_s",  "Decompression Throughput (GB/s)","decompress_time"),
]


def _stream_agg(sdf: pd.DataFrame, algo: str, metric_col: str,
                fmt_filter=None, dk_filter=None) -> pd.DataFrame:
    """
    For a given algorithm and metric, average over splits then compute
    mean+std across variances.

    Returns DataFrame with columns:
      data_key, stream, format, mean, std
    """
    mask = sdf["algorithm"] == algo
    if fmt_filter is not None:
        mask &= sdf["format"] == fmt_filter
    if dk_filter is not None:
        mask &= sdf["data_key"] == dk_filter
    sub = sdf[mask].dropna(subset=[metric_col])
    if sub.empty:
        return pd.DataFrame(columns=["data_key","stream","format","mean","std"])
    # average over splits
    cell = (sub.groupby(["data_key","stream","format","variance"], observed=True)
            [metric_col].mean().reset_index())
    agg = (cell.groupby(["data_key","stream","format"], observed=True)
           [metric_col].agg(["mean","std"]).reset_index())
    agg.columns = ["data_key","stream","format","mean","std"]
    return agg


def _single_bar_plot(ax, x_labels: list, means: list, stds: list,
                     color: str, title: str, ylabel: str, xlabel: str):
    """Draw a simple single-series bar chart with error bars."""
    x = np.arange(len(x_labels))
    ax.bar(x, means, color=color, alpha=0.85, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)


def _grouped_bar_plot(ax, x_labels: list, data: dict,
                      title: str, ylabel: str, xlabel: str,
                      legend_title: str = ""):
    """
    data : OrderedDict  group_label → (means_list, stds_list, color)
    x_labels : list of x-tick labels
    """
    n_groups = len(data)
    n_x      = len(x_labels)
    bar_w    = 0.82 / max(n_groups, 1)
    x        = np.arange(n_x)

    for gi, (glabel, (means, stds, color)) in enumerate(data.items()):
        offsets = x - 0.41 + (gi+0.5)*bar_w
        ax.bar(offsets, means, width=bar_w, color=color,
               label=glabel, alpha=0.85, zorder=3)
        ax.errorbar(offsets, means, yerr=stds, fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    if n_groups > 1:
        ax.legend(title=legend_title,
                  bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, title_fontsize=8)


# ──────────────────────────────────────────────────────────────────────────────
# ⑨ SEMANTIC SEP PLOTS  (3 plots, bars = quantization)
# ──────────────────────────────────────────────────────────────────────────────

# Fixed logical x-axis for semantic_sep (weights then scales):
_SEMANTIC_STREAMS_W = ["sign", "exp", "mant"]
_SEMANTIC_STREAMS_S = ["sign", "exp", "mant"]   # magnitude_index treated as exp

def plot_semantic_sep(sdf: pd.DataFrame, out_dir: Path):
    """
    3 plots for semantic_sep.

    x-axis  : w_sign, w_exp, w_mant, s_sign, s_exp, s_mant
              (scale ticks omitted for formats without scales)
    bars    : one per quantization format (7 colours, low→high precision)
    error bars: std across variances
    """
    algo    = "semantic_sep"
    sub     = sdf[sdf["algorithm"] == algo]
    if sub.empty:
        print("  skip semantic_sep (no stream data)")
        return

    codecs = sorted(sub["codec"].dropna().unique())

    for codec in codecs:
        csub = sub[sub["codec"] == codec]
        fmts_present = [f for f in FORMAT_ORDER_ASC if f in csub["format"].unique()]
        if not fmts_present:
            continue

        # Build x-axis: weights streams for ALL formats, then scale streams
        # only for formats that have scales.
        # We show one tick per (data_key, stream) pair;
        # formats that don't have that tick get no bar (not 0, just absent).
        w_ticks = [("weights", s) for s in _SEMANTIC_STREAMS_W]
        s_ticks = [("scales",  s) for s in _SEMANTIC_STREAMS_S
                   if any(f in SCALED_FORMATS for f in fmts_present)]
        all_ticks   = w_ticks + s_ticks
        tick_labels = (["W: "+s for _,s in w_ticks] +
                       ["S: "+s for _,s in s_ticks])

        for metric_col, metric_label, metric_tag in _STREAM_METRICS:
            # Decompression throughput is only valid for zstd; skip huffman
            if metric_tag == "decompress_time" and codec == "huffman":
                continue
            fig_w = max(8, len(all_ticks)*0.9 + 4)
            fig, ax = plt.subplots(figsize=(fig_w, 5))

            bar_w = 0.82 / max(len(fmts_present), 1)
            x     = np.arange(len(all_ticks))

            for fi, fmt in enumerate(fmts_present):
                color   = _FMT_COLOR_MAP.get(fmt, "#888")
                means   = []
                stds    = []
                for dk, sname in all_ticks:
                    # skip scale ticks for formats without scales
                    if dk == "scales" and fmt not in SCALED_FORMATS:
                        means.append(np.nan); stds.append(0.0)
                        continue
                    # magnitude_index is treated as exp — match both names
                    _snames = ["exp", "magnitude_index"] if sname == "exp" else [sname]
                    fsub = csub[
                        (csub["format"]   == fmt) &
                        (csub["data_key"] == dk) &
                        (csub["stream"].isin(_snames))
                    ].dropna(subset=[metric_col])
                    if fsub.empty:
                        means.append(np.nan); stds.append(0.0)
                    else:
                        cell = (fsub.groupby(["variance"], observed=True)
                                [metric_col].mean())
                        means.append(float(cell.mean()))
                        stds.append(float(cell.std(ddof=1)) if len(cell)>1 else 0.0)

                offsets = x - 0.41 + (fi+0.5)*bar_w
                valid   = [not np.isnan(m) for m in means]
                if any(valid):
                    ax.bar([o for o,v in zip(offsets,valid) if v],
                           [m for m,v in zip(means,valid) if v],
                           width=bar_w, color=color,
                           label=_fmt_label(fmt), alpha=0.85, zorder=3)
                    ax.errorbar(
                        [o for o,v in zip(offsets,valid) if v],
                        [m for m,v in zip(means,valid) if v],
                        yerr=[s for s,v in zip(stds,valid) if v],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
            ax.set_xlabel("Stream  (W = weights, S = scales)", fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.legend(title=LABELS["legends"]["format"],
                      bbox_to_anchor=(1.01,1), loc="upper left",
                      fontsize=7, title_fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
            fig.tight_layout()
            _save(fig, out_dir / f"semantic_sep" / codec / f"{metric_tag}.png")


# ──────────────────────────────────────────────────────────────────────────────
# ⑩ BYTE TRANSPOSE & BITPLANE PLOTS  (per-format × per-codec, single bar)
# ──────────────────────────────────────────────────────────────────────────────

def _per_format_stream_plots(sdf: pd.DataFrame, algo: str, out_dir: Path):
    """
    For byte_transpose or bitplane: one folder per (format × codec).
    Each folder has 3 plots (ratio, compress time, decompress time).
    x-axis  : weight streams in logical order, then scale streams if applicable
              (prefixed W: / S: in labels)
    bars    : single bar per stream, colour = codec colour
    error bars: std across variances
    """
    sub = sdf[sdf["algorithm"] == algo]
    if sub.empty:
        print(f"  skip {algo} (no stream data)")
        return

    codecs       = sorted(sub["codec"].dropna().unique())
    fmts_present = [f for f in FORMAT_ORDER if f in sub["format"].unique()]

    for fmt in fmts_present:
        for codec in codecs:
            fsub = sub[(sub["format"]==fmt)&(sub["codec"]==codec)]
            if fsub.empty:
                continue

            color = _CODEC_COLORS.get(codec, "#5a7fc4")

            # Collect all stream names present for this format
            w_streams = sorted(
                fsub[fsub["data_key"]=="weights"]["stream"].dropna().unique(),
                key=_stream_sort_key)
            s_streams = sorted(
                fsub[fsub["data_key"]=="scales"]["stream"].dropna().unique(),
                key=_stream_sort_key)

            all_ticks   = [("weights",s) for s in w_streams] + \
                          [("scales", s) for s in s_streams]
            tick_labels = ["W:"+s for s in w_streams] + ["S:"+s for s in s_streams]

            if not all_ticks:
                continue

            for metric_col, metric_label, metric_tag in _STREAM_METRICS:
                if metric_tag == "decompress_time" and codec == "huffman":
                    continue
                means = []; stds = []
                for dk, sname in all_ticks:
                    cell = (fsub[(fsub["data_key"]==dk)&(fsub["stream"]==sname)]
                            .dropna(subset=[metric_col])
                            .groupby("variance", observed=True)[metric_col].mean())
                    if cell.empty:
                        means.append(0.0); stds.append(0.0)
                    else:
                        means.append(float(cell.mean()))
                        stds.append(float(cell.std(ddof=1)) if len(cell)>1 else 0.0)

                fig_w = max(6, len(tick_labels)*0.55 + 3)
                fig, ax = plt.subplots(figsize=(fig_w, 5))
                _single_bar_plot(
                    ax, tick_labels, means, stds, color,
                    title=(f"{LABELS['algorithms'].get(algo,algo)} — {_fmt_label(fmt)}\n"
                           f"{metric_label}  {LABELS['codecs'].get(codec,codec)} | "
                           f"{LABELS['bar_subtitle']}"),
                    ylabel=metric_label,
                    xlabel="Stream  (W = weights, S = scales)",
                )
                safe_fmt = fmt.replace("/","_")
                _save(fig, out_dir / f"{algo}_{safe_fmt}" / codec / f"{metric_tag}.png")



# ──────────────────────────────────────────────────────────────────────────────
# VARIANCE HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────

def plot_variance_heatmaps(df: pd.DataFrame, out_dir: Path):
    """
    For each metric (compression ratio, compress throughput, decompress throughput)
    produce two heatmaps saved under out_dir/:
      {metric_tag}_mean.png   — mean across variances
      {metric_tag}_std.png    — std  across variances

    Grid : rows = (algorithm, codec) display labels
           cols = quantization format display labels  (FORMAT_ORDER, high→low)

    Colour scale: white (low) → dark blue (high), shared across the mean and
    std heatmaps of each metric so the scale is directly comparable.

    Decompress throughput: huffman excluded (unmeasured).
    """
    df = df.copy()
    df["total_compress_time_s"]   = df.apply(lambda r: _total_time(r, "compress"),   axis=1)
    df["total_decompress_time_s"] = df.apply(lambda r: _total_time(r, "decompress"), axis=1)
    df["total_original_bytes"] = (
        df["weights_original_bytes"].fillna(0) +
        df["scales_original_bytes"].fillna(0)
    )
    df["compress_throughput_gb_s"]   = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_compress_time_s"]),   axis=1)
    df["decompress_throughput_gb_s"] = df.apply(
        lambda r: _throughput_gb_s(r["total_original_bytes"], r["total_decompress_time_s"]), axis=1)

    metrics = [
        ("weights_ratio",             LABELS["metrics"]["weights_ratio"],          "ratio",          False),
        ("compress_throughput_gb_s",  LABELS["metrics"]["total_compress_time_s"],  "compress_tput",  False),
        ("decompress_throughput_gb_s",LABELS["metrics"]["total_decompress_time_s"],"decompress_tput",True),
    ]

    fmts_present = [f for f in FORMAT_ORDER if f in df["format"].unique()]
    all_fmts     = fmts_present + sorted(
        f for f in df["format"].dropna().unique() if f not in FORMAT_ORDER)
    col_labels   = [_fmt_label(f) for f in all_fmts]

    row_keys = [(algo, codec)
                for algo in ALGORITHM_ORDER
                for codec in sorted(df["codec"].dropna().unique())
                if ((df["algorithm"] == algo) & (df["codec"] == codec)).any()]
    row_labels = [
        f"{LABELS['algorithms'].get(a, a)}\n{LABELS['codecs'].get(c, c)}"
        for a, c in row_keys
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_col, metric_label, metric_tag, exclude_huffman in metrics:
        _df = df if not exclude_huffman else df[df["codec"] != "huffman"]

        # Average over splits first, then compute mean and std over variances
        cell = (
            _df.dropna(subset=[metric_col])
            .groupby(["algorithm", "codec", "format", "variance"], observed=True)
            [metric_col].mean().reset_index()
        )
        agg = (
            cell.groupby(["algorithm", "codec", "format"], observed=True)
            [metric_col].agg(["mean", "std"])
        )
        agg.columns = ["mean", "std"]

        # Build mean and std matrices  (rows × cols)
        n_rows = len(row_keys)
        n_cols = len(all_fmts)
        mean_mat = np.full((n_rows, n_cols), np.nan)
        std_mat  = np.full((n_rows, n_cols), np.nan)

        for ri, (algo, codec) in enumerate(row_keys):
            if exclude_huffman and codec == "huffman":
                continue
            for ci, fmt in enumerate(all_fmts):
                try:
                    r = agg.loc[(algo, codec, fmt)]
                    mean_mat[ri, ci] = r["mean"]
                    std_mat[ri, ci]  = r["std"]
                except KeyError:
                    pass

        # Shared colour scale: vmin=0, vmax = max of BOTH mean and std so that
        # the std heatmap can be directly compared to the mean heatmap.
        vmax_mean = np.nanmax(mean_mat) if not np.all(np.isnan(mean_mat)) else 1.0
        vmax_std  = np.nanmax(std_mat)  if not np.all(np.isnan(std_mat))  else 1.0
        vmax      = max(vmax_mean, vmax_std)

        cmap = plt.cm.Blues

        for stat, mat, fname_tag in [("Mean", mean_mat, "mean"),
                                     ("Std Dev", std_mat,  "std")]:
            fig_h = max(4, n_rows * 0.38 + 1.5)
            fig_w = max(5, n_cols * 0.85 + 2.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            im = ax.imshow(mat, aspect="auto", cmap=cmap,
                           vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                         label=f"{metric_label} ({stat})")

            # Annotate each cell
            for ri in range(n_rows):
                for ci in range(n_cols):
                    val = mat[ri, ci]
                    if np.isnan(val):
                        txt = "—"
                        color = "grey"
                    else:
                        txt   = f"{val:.3f}"
                        # Use white text on dark cells, black on light cells
                        color = "white" if val > 0.55 * vmax else "black"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=6, color=color)

            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=8)
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(row_labels, fontsize=7)

            fig.tight_layout()
            _save(fig, out_dir / f"{metric_tag}_{fname_tag}.png")

# ──────────────────────────────────────────────────────────────────────────────
# ⑪ MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    flat, streams = load_both()
    print(f"  Flat rows  : {len(flat)}")
    print(f"  Stream rows: {len(streams)}")

    print("\n=== Variance table ===")
    make_variance_tables(flat, OUTPUT_DIR)

    print("\n=== Overall bar plots ===")
    plot_algorithm_bars(flat, OUTPUT_DIR / "Overall")

    print("\n=== Variance heatmaps ===")
    plot_variance_heatmaps(flat, OUTPUT_DIR / "Variance")

    if not streams.empty:
        print("\n=== Semantic sep stream plots ===")
        plot_semantic_sep(streams, OUTPUT_DIR)

        print("\n=== Byte-transpose per-format stream plots ===")
        _per_format_stream_plots(streams, "byte_transpose", OUTPUT_DIR)

        print("\n=== Bitplane per-format stream plots ===")
        _per_format_stream_plots(streams, "bitplane", OUTPUT_DIR)
    else:
        print("\n  (no per-stream data found — skipping stream plots)")

    print(f"\nDone. All outputs under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()