import argparse
from pathlib import Path

import numpy as np
import scvi  # type: ignore
from anndata import read_h5ad  # type: ignore


def log(msg: str) -> None:
    print(f"[scvi] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scVI on an AnnData file.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--tissue", required=True)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu", help="Trainer accelerator, e.g. cpu, mps, gpu.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices for the trainer.")
    args = parser.parse_args()

    log(f"Loading {args.input_file}")
    adata = read_h5ad(args.input_file)
    log(f"AnnData shape: {adata.shape}")

    scvi.settings.seed = args.seed
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=args.latent_dim)
    log("Training scVI...")
    model.train(max_epochs=args.max_epochs, accelerator=args.accelerator, devices=args.devices)
    log("Training complete, exporting normalized expression")

    imputed = model.get_normalized_expression(return_numpy=True)
    output_dir = Path(args.output_root) / "scVI" / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.tissue}.npy"
    np.save(output_path, imputed.astype(np.float32))
    log(f"Saved scVI output to {output_path}")


if __name__ == "__main__":
    main()
