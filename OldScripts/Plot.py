#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_JSONL = "compression_results_huffman.jsonl"
OUTPUT_DIR = Path("PLOTS")

# Only splits and levels that the current benchmark actually produces
SPLITS_TO_PLOT    = ["Base", "XOR_Delta"]
ZSTD_LEVELS_TO_PLOT = [1]

FORMAT_ORDER = [
    "FP32E8M23",
    "BF16E8M7",
    "FP16E5M10",
    "FP8E5M2",
    "FP8E4M3",
    "MXFP4E2M1",
    "NVFP4E2M1",
]

METRICS = {
    "weights_ratio":             "Compression Ratio",
    "weights_compress_time_s":   "Compression Time (s)",
    "weights_decompress_time_s": "Decompression Time (s)",
}


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

            weights = obj.get("weights", {}) or {}
            scales  = obj.get("scales",  {}) or {}

            rows.append({
                "format":                    obj.get("format"),
                "variance":                  obj.get("variance"),
                "split":                     obj.get("split"),
                "algorithm":                 obj.get("algorithm"),
                "zstd_level":                obj.get("zstd_level"),
                "weights_ratio":             weights.get("ratio"),
                "weights_compress_time_s":   weights.get("compress_time_s"),
                "weights_decompress_time_s": weights.get("decompress_time_s"),
                "scales_ratio":              scales.get("ratio"),
                "scales_compress_time_s":    scales.get("compress_time_s"),
                "scales_decompress_time_s":  scales.get("decompress_time_s"),
            })

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return pd.DataFrame(rows)


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
        temp["variance"]
        .astype(str)
        .str.replace("Var", "", regex=False)
        .str.replace("var", "", regex=False)
        .str.replace("p",   ".", regex=False)
    )
    temp["variance_num"] = pd.to_numeric(temp["variance_num"], errors="coerce")
    if temp["variance_num"].notna().all():
        temp = temp.sort_values("variance_num")
        return temp["variance"].tolist()
    return raw_values


def make_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_grouped_bars(
    df: pd.DataFrame,
    split_name: str,
    zstd_level: int,
    metric_col: str,
    metric_label: str,
    group_col: str,
    group_label: str,
    algo_col: str,
    output_path: Path,
):
    sub = df[
        (df["split"]      == split_name)
        & (df["zstd_level"] == zstd_level)
        & df[metric_col].notna()
        & df[group_col].notna()
        & df[algo_col].notna()
    ].copy()

    if sub.empty:
        print(
            f"Skipping empty plot: split={split_name}, zstd={zstd_level}, "
            f"metric={metric_col}, group={group_col}"
        )
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
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index=group_col, columns=algo_col, values=metric_col)
    pivot = pivot.reindex(index=group_values, columns=algo_values)
    pivot = pivot.dropna(axis=0, how="all")
    pivot = pivot.dropna(axis=1, how="all")

    if pivot.empty:
        print(
            f"Skipping fully empty pivot: split={split_name}, zstd={zstd_level}, "
            f"metric={metric_col}, group={group_col}"
        )
        return

    n_groups  = len(pivot.index)
    n_algos   = len(pivot.columns)
    x         = np.arange(n_groups)
    total_width = 0.84
    bar_width   = total_width / max(n_algos, 1)

    fig_width = max(12, 1.6 * n_groups + 4)
    fig, ax   = plt.subplots(figsize=(fig_width, 7))

    for i, algo in enumerate(pivot.columns):
        offsets = x - (total_width / 2) + (i + 0.5) * bar_width
        ax.bar(offsets, pivot[algo].values, width=bar_width, label=str(algo))

    ax.set_title(
        f"{metric_label} by {group_label} | Split={split_name} | zstd_level={zstd_level}"
    )
    ax.set_xlabel(group_label)
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in pivot.index], rotation=30, ha="right")
    ax.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    make_output_dir()

    df = load_jsonl(INPUT_JSONL)

    required_columns = ["format", "variance", "split", "algorithm", "zstd_level"]
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")

    missing_metrics = [m for m in METRICS if m not in df.columns]
    if missing_metrics:
        raise KeyError(
            f"Missing required metric columns: {missing_metrics}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = coerce_numeric(df, ["zstd_level"] + list(METRICS.keys()))

    present_formats = [f for f in FORMAT_ORDER if f in set(df["format"].astype(str))]
    df["format"] = pd.Categorical(df["format"], categories=present_formats, ordered=True)

    variance_order = sort_variance_values(df["variance"])
    df["variance"] = pd.Categorical(df["variance"], categories=variance_order, ordered=True)

    for split_name in SPLITS_TO_PLOT:
        for zstd_level in ZSTD_LEVELS_TO_PLOT:
            for metric_col, metric_label in METRICS.items():

                plot_grouped_bars(
                    df=df,
                    split_name=split_name,
                    zstd_level=zstd_level,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    group_col="format",
                    group_label="Quantization Type / Format",
                    algo_col="algorithm",
                    output_path=OUTPUT_DIR / f"{metric_col}_by_format_split_{split_name}_zstd{zstd_level}.png",
                )

                plot_grouped_bars(
                    df=df,
                    split_name=split_name,
                    zstd_level=zstd_level,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    group_col="variance",
                    group_label="Variance",
                    algo_col="algorithm",
                    output_path=OUTPUT_DIR / f"{metric_col}_by_variance_split_{split_name}_zstd{zstd_level}.png",
                )

    print("\nDone.")
    print(f"Plots saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()