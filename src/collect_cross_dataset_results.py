#!/usr/bin/env python3
"""Combine the PathMNIST and CelebA sigma/CFG sweep results."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pathmnist-csv",
        default=(
            "/content/drive/MyDrive/SRUL_Final_Comparisons/"
            "PathMNIST_knob_cfg/seed_0/knob_cfg_sweep_metrics.csv"
        ),
    )
    p.add_argument(
        "--celeba-csv",
        default=(
            "/content/drive/MyDrive/SRUL_Final_Comparisons/"
            "CelebA64_knob_cfg/seed_0/knob_cfg_sweep_metrics.csv"
        ),
    )
    p.add_argument(
        "--output",
        default=(
            "/content/drive/MyDrive/SRUL_Final_Comparisons/"
            "cross_dataset_knob_cfg_results.csv"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for dataset, filename in [
        ("PathMNIST", args.pathmnist_csv),
        ("CelebA-64", args.celeba_csv),
    ]:
        path = Path(filename)
        if not path.exists():
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["dataset_display"] = dataset
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No sweep CSV was found.")

    result = pd.concat(frames, ignore_index=True)
    preferred = [
        "dataset_display",
        "sigma_enc",
        "guidance_scale",
        "reconstruction_fid",
        "fid",
        "kid_mean",
        "feature_precision",
        "feature_recall",
        "prior_final_loss",
    ]
    columns = [c for c in preferred if c in result.columns]
    result = result[columns].sort_values(
        ["dataset_display", "sigma_enc", "guidance_scale"]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print("\nAll results:\n")
    print(result.to_string(index=False))

    if "fid" in result.columns:
        best = result.loc[result.groupby("dataset_display")["fid"].idxmin()]
        print("\nBest FID setting per dataset:\n")
        print(best.to_string(index=False))

    print(f"\nSaved combined table to: {output}")


if __name__ == "__main__":
    main()
