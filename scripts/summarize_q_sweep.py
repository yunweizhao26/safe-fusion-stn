import argparse
import csv
import glob
import re


def pick_row(rows, method, fill):
    for row in rows:
        if row["method"] != method:
            continue
        if abs(float(row["fill_rate_target"]) - fill) < 1e-9:
            return row
    return None


def main():
    ap = argparse.ArgumentParser(description="Summarize q-sweep at a fixed fill rate.")
    ap.add_argument("--glob", required=True, help="Glob for q-sweep method_operating_points.csv files")
    ap.add_argument("--fill", type=float, default=0.081)
    ap.add_argument("--out", type=str, default="", help="Optional CSV output path")
    args = ap.parse_args()

    rows_out = []
    for path in sorted(glob.glob(args.glob)):
        if "_seed" in path:
            continue
        match = re.search(r"_q([^/]+)", path)
        if not match:
            continue
        qtag = match.group(1)
        q = float(qtag.replace("p", "."))
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        safe = pick_row(rows, "safe_latent_truth_scvi_select", args.fill)
        scvi = pick_row(rows, "scVI", args.fill)
        if not safe or not scvi:
            continue
        safe_f1 = float(safe["f1"])
        scvi_f1 = float(scvi["f1"])
        rows_out.append(
            {
                "q": q,
                "safe_f1": safe_f1,
                "scvi_f1": scvi_f1,
                "gap": safe_f1 - scvi_f1,
                "safe_precision": float(safe["precision"]),
                "safe_recall": float(safe["recall"]),
                "coverage_fusion": float(safe.get("coverage_fusion") or "nan"),
                "rare_support": float(safe.get("rare_support") or "nan"),
            }
        )

    rows_out.sort(key=lambda row: row["q"])
    if not rows_out:
        raise SystemExit("No q-sweep rows found.")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=list(rows_out[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    import sys

    main()
