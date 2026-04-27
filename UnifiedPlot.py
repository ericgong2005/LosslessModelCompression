#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

INPUT_JSONL = "compression_results_huffman_finegrained_2_10.jsonl"   # change to zstd file as needed

# Output dir is derived automatically from the filename
_stem = Path(INPUT_JSONL).stem   # e.g. "compression_results_huffman"
if "huffman" in _stem:
    OUTPUT_DIR = Path("PLOTS_HUFFMAN_FineGrained")
    CODEC_LABEL = "Huffman"
else:
    OUTPUT_DIR = Path("PLOTS_ZSTD_FineGrained")
    CODEC_LABEL = "Zstd"

SPLITS_TO_PLOT = ["Base", "XOR_Delta"]

FORMAT_ORDER = [
    "FP32E8M23",
    "BF16E8M7",
    "FP16E5M10",
    "FP8E5M2",
    "FP8E4M3",
    "MXFP4E2M1",
    "NVFP4E2M1",
]

# Top-level aggregate metrics (averaged over variances for the summary plots)
AGGREGATE_METRICS = {
    "weights_ratio":             "Compression Ratio (weights)",
    "weights_compress_time_s":   "Compression Time s (weights)",
    "weights_decompress_time_s": "Decompression Time s (weights)",
}

# ──────────────────────────────────────────────────────────────────────────────
# JSONL loader
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    """
    Load the JSONL into a flat DataFrame for aggregate metrics, plus return
    the raw list of records for per-stream plots.
    """
    rows    = []
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            records.append(obj)

            weights = obj.get("weights", {}) or {}
            scales  = obj.get("scales",  {}) or {}

            rows.append({
                "format":                    obj.get("format"),
                "variance":                  obj.get("variance"),
                "split":                     obj.get("split"),
                "algorithm":                 obj.get("algorithm"),
                "weights_ratio":             weights.get("ratio"),
                "weights_compress_time_s":   weights.get("compress_time_s"),
                "weights_decompress_time_s": weights.get("decompress_time_s"),
                "scales_ratio":              scales.get("ratio"),
                "scales_compress_time_s":    scales.get("compress_time_s"),
                "scales_decompress_time_s":  scales.get("decompress_time_s"),
            })

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return pd.DataFrame(rows), records


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def coerce_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sort_variance_values(series: pd.Series):
    raw_values = [v for v in pd.unique(series) if pd.notna(v)]
    temp = pd.DataFrame({"variance": raw_values})
    temp["variance_num"] = (
        temp["variance"].astype(str)
        .str.replace("Var", "", regex=False)
        .str.replace("var", "", regex=False)
        .str.replace("p",   ".", regex=False)
    )
    temp["variance_num"] = pd.to_numeric(temp["variance_num"], errors="coerce")
    if temp["variance_num"].notna().all():
        temp = temp.sort_values("variance_num")
        return temp["variance"].tolist()
    return raw_values


def make_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Summary / aggregate plots  (existing behaviour, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def plot_grouped_bars(
    df, split_name, metric_col, metric_label, group_col, group_label,
    algo_col, output_path,
):
    sub = df[
        (df["split"] == split_name)
        & df[metric_col].notna()
        & df[group_col].notna()
        & df[algo_col].notna()
    ].copy()

    if sub.empty:
        print(f"  skip (empty): {output_path.name}")
        return

    if group_col == "format":
        group_values = [g for g in FORMAT_ORDER if g in set(sub["format"].astype(str))]
    else:
        group_values = [
            g for g in sub[group_col].cat.categories
            if g in set(sub[group_col].astype(str))
        ]

    algo_values = sorted(pd.unique(sub[algo_col].astype(str)).tolist())
    grouped = (
        sub.groupby([group_col, algo_col], observed=False)[metric_col]
        .mean().reset_index()
    )
    pivot = grouped.pivot(index=group_col, columns=algo_col, values=metric_col)
    pivot = pivot.reindex(index=group_values, columns=algo_values)
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if pivot.empty:
        print(f"  skip (all-NaN): {output_path.name}")
        return

    n_groups    = len(pivot.index)
    n_algos     = len(pivot.columns)
    x           = np.arange(n_groups)
    total_width = 0.84
    bar_width   = total_width / max(n_algos, 1)

    fig, ax = plt.subplots(figsize=(max(12, 1.6 * n_groups + 4), 7))
    for i, algo in enumerate(pivot.columns):
        offsets = x - (total_width / 2) + (i + 0.5) * bar_width
        ax.bar(offsets, pivot[algo].values, width=bar_width, label=str(algo))

    ax.set_title(f"{metric_label} by {group_label} | Split={split_name} | {CODEC_LABEL}")
    ax.set_xlabel(group_label)
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index], rotation=30, ha="right")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-stream plots
#
# For a given (algorithm, format) pair, collect all records across variances
# and splits, then for each stream plot:
#   • compression ratio  (compressed_bytes / original_bytes)  — bar per stream
#   • compress_time_s    — bar per stream
#   • decompress_time_s  — bar per stream
# averaged over variances, for weights and scales separately.
#
# Layout: one figure per metric, one subplot per split.
# Saved under OUTPUT_DIR / algo / {fmt}_{metric}_{weights|scales}.png
# ──────────────────────────────────────────────────────────────────────────────

def _stream_df_from_records(records, algo, fmt_name, data_key):
    """
    Build a DataFrame of per-stream metrics for one (algo, format, weights|scales).

    Columns: variance, split, stream_name, original_bytes, compressed_bytes,
             ratio, compress_time_s, decompress_time_s
    """
    rows = []
    for rec in records:
        if rec.get("algorithm") != algo:
            continue
        if rec.get("format") != fmt_name:
            continue
        block = rec.get(data_key, {}) or {}
        streams = block.get("streams", {}) or {}
        if not streams:
            continue
        for s_name, s_val in streams.items():
            if not isinstance(s_val, dict):
                continue
            rows.append({
                "variance":          rec.get("variance"),
                "split":             rec.get("split"),
                "stream":            s_name,
                "original_bytes":    s_val.get("original_bytes"),
                "compressed_bytes":  s_val.get("compressed_bytes"),
                "ratio":             s_val.get("ratio"),
                "compress_time_s":   s_val.get("compress_time_s"),
                "decompress_time_s": s_val.get("decompress_time_s"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ["original_bytes","compressed_bytes","ratio",
              "compress_time_s","decompress_time_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ordered_streams(stream_names):
    """
    Return stream names in a sensible display order.

    sign first, then exp bits MSB→LSB, then mant bits MSB→LSB,
    then byte planes b0…bN, then bit planes bit31…bit0, then 'data', then rest.
    """
    def _key(s):
        if s == "sign":       return (0, 0)
        if s == "exp":        return (1, 0)
        if s.startswith("exp_bit"):
            try:   return (1, -int(s[7:]))
            except: pass
        if s.startswith("mag_bit"):
            try:   return (2, -int(s[7:]))
            except: pass
        if s.startswith("mant_bit"):
            try:   return (3, -int(s[8:]))
            except: pass
        if s.startswith("b") and s[1:].isdigit():
            return (4, int(s[1:]))
        if s.startswith("bit") and s[3:].isdigit():
            return (5, -int(s[3:]))
        if s == "data":       return (6, 0)
        return (7, 0)
    return sorted(stream_names, key=_key)


def plot_per_stream(records, algo, fmt_name, out_dir):
    """
    Generate per-stream plots for one (algorithm, format) pair.
    Produces up to 6 PNGs (3 metrics × 2 data keys) if data is available.
    """
    STREAM_METRICS = {
        "ratio":             "Compression Ratio",
        "compress_time_s":   "Compress Time (s)",
        "decompress_time_s": "Decompress Time (s)",
    }

    for data_key in ("weights", "scales"):
        sdf = _stream_df_from_records(records, algo, fmt_name, data_key)
        if sdf.empty:
            continue

        # Only keep splits that appear in data
        splits_present = [s for s in SPLITS_TO_PLOT if s in sdf["split"].unique()]
        if not splits_present:
            continue

        stream_order = _ordered_streams(sdf["stream"].unique().tolist())

        for metric, metric_label in STREAM_METRICS.items():
            # One subplot per split, side by side
            n_splits = len(splits_present)
            fig, axes = plt.subplots(
                1, n_splits,
                figsize=(max(8, len(stream_order) * 0.55 + 3) * n_splits, 5),
                sharey=True, squeeze=False,
            )
            fig.suptitle(
                f"{CODEC_LABEL} | {algo} | {fmt_name} | {data_key} | {metric_label}",
                fontsize=11,
            )

            for ax, split_name in zip(axes[0], splits_present):
                sub = sdf[sdf["split"] == split_name]
                if sub.empty or sub[metric].isna().all():
                    ax.set_title(f"Split={split_name}  (no data)")
                    ax.set_xticks([])
                    continue

                # Average over variances
                agg = (
                    sub.groupby("stream", observed=False)[metric]
                    .mean()
                    .reindex(stream_order)
                )
                x      = np.arange(len(stream_order))
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

                bars = ax.bar(x, agg.values, color=[
                    colors[i % len(colors)] for i in range(len(stream_order))
                ])
                ax.set_title(f"Split={split_name}")
                ax.set_xlabel("Stream")
                ax.set_ylabel(metric_label if split_name == splits_present[0] else "")
                ax.set_xticks(x)
                ax.set_xticklabels(stream_order, rotation=60, ha="right", fontsize=7)
                ax.grid(axis="y", linestyle="--", alpha=0.35)

                # Annotate bars with value
                for bar, val in zip(bars, agg.values):
                    if pd.isna(val):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=5, rotation=45,
                    )

            fig.tight_layout()
            fname = f"{fmt_name}_{data_key}_{metric}.png"
            out_path = out_dir / fname
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    make_dir(OUTPUT_DIR)

    df, records = load_jsonl(INPUT_JSONL)

    required = ["format", "variance", "split", "algorithm"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = coerce_numeric(df, list(AGGREGATE_METRICS.keys()))

    present_formats = [f for f in FORMAT_ORDER if f in set(df["format"].astype(str))]
    df["format"] = pd.Categorical(df["format"], categories=present_formats, ordered=True)

    variance_order = sort_variance_values(df["variance"])
    df["variance"] = pd.Categorical(df["variance"], categories=variance_order, ordered=True)

    # ── 1. Summary / aggregate plots (existing) ───────────────────────────────
    summary_dir = OUTPUT_DIR / "summary"
    make_dir(summary_dir)
    print(f"\n=== Summary plots → {summary_dir} ===")

    for split_name in SPLITS_TO_PLOT:
        for metric_col, metric_label in AGGREGATE_METRICS.items():
            plot_grouped_bars(
                df=df, split_name=split_name,
                metric_col=metric_col, metric_label=metric_label,
                group_col="format", group_label="Quantization Format",
                algo_col="algorithm",
                output_path=summary_dir / f"{metric_col}_by_format_split_{split_name}.png",
            )
            plot_grouped_bars(
                df=df, split_name=split_name,
                metric_col=metric_col, metric_label=metric_label,
                group_col="variance", group_label="Variance",
                algo_col="algorithm",
                output_path=summary_dir / f"{metric_col}_by_variance_split_{split_name}.png",
            )

    # ── 2. Per-stream plots: OUTPUT_DIR / {algo} / {fmt}_{data_key}_{metric}.png
    algos_in_data  = df["algorithm"].dropna().unique().tolist()
    formats_in_data = df["format"].dropna().astype(str).unique().tolist()

    print(f"\n=== Per-stream plots ===")
    for algo in algos_in_data:
        algo_dir = OUTPUT_DIR / algo
        make_dir(algo_dir)
        print(f"\n  Algorithm: {algo}  → {algo_dir}")
        for fmt_name in [f for f in FORMAT_ORDER if f in formats_in_data]:
            plot_per_stream(records, algo, fmt_name, algo_dir)

    print(f"\nDone. All plots under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()