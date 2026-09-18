#!/usr/bin/env python3
"""
extract_results.py

Extracts and summarizes results from a Hy2DL testing run.
Reads testing_metrics.zarr and testing_results.zarr from a run folder,
prints a summary of the per-basin metrics, and saves them to CSV.

Usage:
    python extract_results.py
    python extract_results.py --run_dir results/LSTM_CAMELS_DE_1h_benchmark_seed_100
    python extract_results.py --run_dir results/LSTM_CAMELS_DE_1h_benchmark_seed_100 --metric NSE --top_n 20

Run this from inside the Hy2DL-Testing directory (or pass the full path with --run_dir).
"""

import argparse
import sys
from pathlib import Path

import xarray as xr


def load_zarr(path: Path, label: str):
    if not path.exists():
        print(f"[!] {label} not found at: {path}")
        return None
    try:
        ds = xr.open_zarr(path)
        print(f"\n=== {label} ===")
        print(ds)
        return ds
    except Exception as e:
        print(f"[!] Failed to open {label} at {path}: {e}")
        return None


def summarize_metrics(ds_metrics, metric: str, top_n: int):
    if ds_metrics is None:
        return None

    df = ds_metrics.to_dataframe()

    print(f"\n=== Summary across all basins ({df.shape[0]} rows) ===")
    print(df.describe())

    if metric not in df.columns:
        print(f"\n[!] Metric '{metric}' not found. Available columns: {list(df.columns)}")
        return df

    print(f"\n=== Top {top_n} basins by {metric} ===")
    print(df.sort_values(metric, ascending=False).head(top_n))

    print(f"\n=== Bottom {top_n} basins by {metric} ===")
    print(df.sort_values(metric, ascending=True).head(top_n))

    print(f"\n{metric} median: {df[metric].median():.4f}")
    print(f"{metric} mean:   {df[metric].mean():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Summarize Hy2DL testing results.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="results/LSTM_CAMELS_DE_1h_benchmark_seed_100",
        help="Path to the run folder containing the .zarr stores (relative or absolute).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="NSE",
        help="Metric column to rank basins by (e.g. NSE, KGE, RMSE). Default: NSE",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of best/worst basins to display. Default: 20",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional path to save the per-basin metrics as CSV. "
             "Defaults to <run_dir>/metrics_summary.csv",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "testing_metrics.zarr"
    results_path = run_dir / "testing_results.zarr"

    if not run_dir.exists():
        print(f"[!] Run directory does not exist: {run_dir.resolve()}")
        sys.exit(1)

    ds_metrics = load_zarr(metrics_path, "testing_metrics.zarr")
    df_metrics = summarize_metrics(ds_metrics, args.metric, args.top_n)

    # Load testing_results.zarr just to show its structure (predictions/observations).
    # Not loaded into memory as a dataframe since it's typically much larger
    # (per-basin, per-timestep discharge series).
    load_zarr(results_path, "testing_results.zarr")

    if df_metrics is not None:
        out_csv = Path(args.out_csv) if args.out_csv else run_dir / "metrics_summary.csv"
        df_metrics.to_csv(out_csv)
        print(f"\n[+] Saved per-basin metrics to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()