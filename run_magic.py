import argparse
from pathlib import Path

import anndata as ad  # type: ignore
import magic  # type: ignore
import numpy as np


def log(msg: str) -> None:
    print(f"[magic] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAGIC imputation on an AnnData file.")
    parser.add_argument("--input_file", required=True, help="Path to .h5ad file.")
    parser.add_argument("--output_root", required=True, help="Destination root directory (e.g., ep_dataset/GSE100866/imputed).")
    parser.add_argument("--dataset_id", required=True, help="Dataset identifier (used for folder naming).")
    parser.add_argument("--tissue", required=True, help="Tissue/sample name (used for filename).")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed for MAGIC.")
    parser.add_argument("--knn", type=int, default=15, help="Number of neighbors for MAGIC.")
    args = parser.parse_args()

    log(f"Loading {args.input_file}")
    adata = ad.read_h5ad(args.input_file)
    log(f"AnnData shape: {adata.shape}")
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    magic_op = magic.MAGIC(knn=args.knn, random_state=args.random_state, n_jobs=1)
    log("Running MAGIC...")
    imputed = magic_op.fit_transform(X)
    log("MAGIC completed.")

    output_dir = Path(args.output_root) / "MAGIC" / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.tissue}.npy"
    np.save(output_path, imputed.astype(np.float32))
    log(f"Saved imputed matrix to {output_path}")


if __name__ == "__main__":
    main()
