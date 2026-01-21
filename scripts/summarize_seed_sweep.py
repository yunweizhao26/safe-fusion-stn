import argparse
import csv
import glob


def load_rows(path: str, method: str, fill: float):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            if abs(float(row["fill_rate_target"]) - fill) < 1e-9:
                rows.append(row)
                break
    return rows


def summarize(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    std = var ** 0.5
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize seed sweeps at a fixed fill rate.")
    ap.add_argument("--glob", required=True, help="Glob for method_operating_points.csv files")
    ap.add_argument("--fill", type=float, default=0.081)
    ap.add_argument("--methods", required=True, help="Comma-separated methods to summarize")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit("No files matched the glob.")

    rows = []
    for method in methods:
        for metric in ("precision", "recall", "f1", "coverage_fusion", "rare_support"):
            vals = []
            for path in files:
                for row in load_rows(path, method, args.fill):
                    val = row.get(metric, "")
                    if val != "":
                        vals.append(float(val))
            stats = summarize(vals)
            if not stats:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    **stats,
                }
            )

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["method", "metric", "mean", "std", "min", "max", "n"],
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    import sys

    main()
