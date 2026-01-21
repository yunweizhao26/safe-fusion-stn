import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot F1 vs fill-rate for selected methods.")
    ap.add_argument("--csv", required=True, help="Path to method_operating_points.csv")
    ap.add_argument("--methods", required=True, help="Comma-separated method names")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    plt.figure(figsize=(4.2, 3.2))
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        plt.plot(sub["fill_rate_target"], sub["f1"], marker="o", label=method)

    plt.xlabel("fill rate")
    plt.ylabel("F1")
    if args.title:
        plt.title(args.title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
