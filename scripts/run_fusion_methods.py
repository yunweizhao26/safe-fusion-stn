import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Ensure repo root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA

from fusion.data import (
    compute_teacher_from_counts,
    infer_dataset_id,
    load_counts,
    load_teacher_matrices,
    save_imputed_matrix,
    save_metadata,
)
from fusion.eval import (
    masked_reconstruction_metrics,
    predict_diffusion,
    predict_latent_truth,
    predict_mixture_of_experts,
    sample_mask,
)
from fusion.train import (
    TrainConfig,
    train_diffusion,
    train_latent_truth,
    train_mixture_of_experts,
)


def log(msg: str) -> None:
    print(f"[fusion] {msg}")


def save_metrics(output_dir: Path, metrics: Dict[str, float], name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    log(f"Saved metrics to {path}")


def log1p_cpm_dense(matrix: np.ndarray) -> np.ndarray:
    libsize = np.sum(matrix, axis=1)
    scale = 1e4 / (libsize + 1e-8)
    scaled = matrix * scale[:, None]
    return np.log1p(scaled)


def build_pca_features(
    train_counts: np.ndarray,
    predict_counts: np.ndarray,
    mask_fraction: float,
    n_components: int,
    seed: int,
) -> tuple:
    rng = np.random.default_rng(seed)
    masked = train_counts.copy()
    nonzero = masked > 0
    mask = (rng.random(masked.shape) < mask_fraction) & nonzero
    masked[mask] = 0.0
    train_norm = log1p_cpm_dense(masked)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    train_pca = pca.fit_transform(train_norm).astype(np.float32)
    predict_norm = log1p_cpm_dense(predict_counts)
    predict_pca = pca.transform(predict_norm).astype(np.float32)
    return pca, train_pca, predict_pca


def build_eval_teachers(
    masked_counts: np.ndarray,
    predict_teachers: Dict[str, np.ndarray],
    teacher_methods: list,
    mask: np.ndarray,
    strategy: str,
    svd_components: int,
    knn_neighbors: int,
    graph_neighbors: int,
    graph_pca_components: int,
    graph_include_self: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Build teacher inputs for masked evaluation.
    strategy:
        - mask: zero masked entries in teacher outputs.
        - recompute: recompute teacher outputs on masked counts.
        - auto: recompute if supported, else mask.
    """
    eval_teachers: Dict[str, np.ndarray] = {}
    for method in teacher_methods:
        if strategy in ("recompute", "auto"):
            try:
                eval_teachers[method] = compute_teacher_from_counts(
                    method,
                    masked_counts,
                    svd_components=svd_components,
                    knn_neighbors=knn_neighbors,
                    graph_neighbors=graph_neighbors,
                    graph_pca_components=graph_pca_components,
                    graph_include_self=graph_include_self,
                    seed=seed,
                )
                continue
            except ValueError:
                if strategy == "recompute":
                    raise
        if method not in predict_teachers:
            raise KeyError(f"Missing teacher '{method}' for evaluation.")
        teacher_matrix = predict_teachers[method].copy()
        teacher_matrix[mask] = 0.0
        eval_teachers[method] = teacher_matrix
    return eval_teachers


def write_sanity_samples(
    output_dir: Path,
    name: str,
    counts: np.ndarray,
    masked_counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    preds: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    seed: int,
) -> None:
    if n_samples <= 0:
        return
    rng = np.random.default_rng(seed)
    indices = np.column_stack(np.where(mask))
    if indices.size == 0:
        return
    sample_n = min(n_samples, indices.shape[0])
    sample_idx = rng.choice(indices.shape[0], size=sample_n, replace=False)
    records = []
    for idx in sample_idx:
        row, col = indices[idx]
        record = {
            "row": int(row),
            "col": int(col),
            "x_true": float(counts[row, col]),
            "x_masked": float(masked_counts[row, col]),
            "pred": float(preds[row, col]),
            "teachers": {name: float(mat[row, col]) for name, mat in teachers.items()},
        }
        records.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sanity_samples_{name}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    log(f"Saved sanity samples to {path}")


def run_latent_truth(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    cfg: TrainConfig,
    output_root: str,
    dataset_id: str,
    disease: str,
    tissue: str,
    seed: int,
    subset_info: Dict[str, int],
    predict_counts: np.ndarray,
    predict_teachers: Dict[str, np.ndarray],
    eval_mask: np.ndarray,
    eval_strategy: str,
    eval_svd_components: int,
    eval_knn_neighbors: int,
    eval_graph_neighbors: int,
    eval_graph_pca_components: int,
    eval_graph_include_self: bool,
    eval_debug_samples: int,
    predict_loglib: np.ndarray,
    gene_mean: np.ndarray,
    gene_dropout: np.ndarray,
    pca_model: PCA = None,
    train_pca: np.ndarray = None,
    predict_pca: np.ndarray = None,
    pca_components: int = 0,
    pca_proj_dim: int = 8,
):
    teacher_mask = None
    if cfg.teacher_likelihood_exclude:
        teacher_mask = np.ones(len(predict_teachers), dtype=np.float32)
        for idx, name in enumerate(predict_teachers.keys()):
            if name in cfg.teacher_likelihood_exclude:
                teacher_mask[idx] = 0.0
    log("Training Latent Truth Fusion model")
    model, history = train_latent_truth(
        counts,
        teachers,
        cfg,
        gene_mean=gene_mean,
        gene_dropout=gene_dropout,
        pca_features=train_pca,
        pca_proj_dim=pca_proj_dim,
        seed=seed,
    )
    mu, pi, var_post = predict_latent_truth(
        model,
        predict_counts,
        teachers=predict_teachers,
        fuse=True,
        cell_loglib=predict_loglib,
        gene_mean=gene_mean,
        gene_dropout=gene_dropout,
        pca_features=predict_pca,
        teacher_mask=teacher_mask,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    save_imputed_matrix(mu, output_root, "latent_truth", dataset_id, disease, tissue)
    pi_path = Path(output_root) / "latent_truth" / dataset_id / disease / f"{tissue}_pi.npy"
    np.save(pi_path, pi.astype(np.float32))
    var_path = Path(output_root) / "latent_truth" / dataset_id / disease / f"{tissue}_var.npy"
    np.save(var_path, var_post.astype(np.float32))
    masked_counts = predict_counts.copy()
    masked_counts[eval_mask] = 0.0
    if not np.allclose(masked_counts[eval_mask], 0.0):
        raise RuntimeError("Masked inputs still contain non-zero entries.")
    masked_loglib = np.log1p(np.sum(masked_counts, axis=1)).astype(np.float32)
    masked_pca = None
    if pca_model is not None:
        masked_norm = log1p_cpm_dense(masked_counts)
        masked_pca = pca_model.transform(masked_norm).astype(np.float32)
    eval_teachers = build_eval_teachers(
        masked_counts,
        predict_teachers,
        list(predict_teachers.keys()),
        eval_mask,
        eval_strategy,
        eval_svd_components,
        eval_knn_neighbors,
        eval_graph_neighbors,
        eval_graph_pca_components,
        eval_graph_include_self,
        seed,
    )
    eval_teacher_mask = None
    if cfg.teacher_likelihood_exclude:
        eval_teacher_mask = np.ones(len(eval_teachers), dtype=np.float32)
        for idx, name in enumerate(eval_teachers.keys()):
            if name in cfg.teacher_likelihood_exclude:
                eval_teacher_mask[idx] = 0.0
    masked_mu, _, _ = predict_latent_truth(
        model,
        masked_counts,
        teachers=eval_teachers,
        fuse=True,
        cell_loglib=masked_loglib,
        gene_mean=gene_mean,
        gene_dropout=gene_dropout,
        pca_features=masked_pca,
        teacher_mask=eval_teacher_mask,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    metrics = masked_reconstruction_metrics(
        predict_counts,
        masked_mu,
        mask_fraction=cfg.mask_fraction,
        seed=seed,
        mask=eval_mask,
    )
    output_dir = Path(output_root) / "latent_truth" / dataset_id / disease
    save_metrics(output_dir, metrics, "masked_metrics.json")
    write_sanity_samples(
        output_dir,
        "masked",
        predict_counts,
        masked_counts,
        eval_teachers,
        masked_mu,
        eval_mask,
        eval_debug_samples,
        seed,
    )
    save_metadata(
        Path(output_root) / "latent_truth" / dataset_id / disease,
        {
            "history": history,
            "config": cfg.__dict__,
            "subset": subset_info,
            "pca_components": pca_components,
            "pca_proj_dim": pca_proj_dim,
        },
    )


def run_mixture_of_experts(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    cfg: TrainConfig,
    output_root: str,
    dataset_id: str,
    disease: str,
    tissue: str,
    seed: int,
    subset_info: Dict[str, int],
    predict_counts: np.ndarray,
    predict_teachers: Dict[str, np.ndarray],
    eval_mask: np.ndarray,
    eval_strategy: str,
    eval_svd_components: int,
    eval_knn_neighbors: int,
    eval_graph_neighbors: int,
    eval_graph_pca_components: int,
    eval_graph_include_self: bool,
    eval_debug_samples: int,
):
    log("Training Mixture-of-Experts model")
    model, history = train_mixture_of_experts(counts, teachers, cfg, seed=seed)
    mu, pi = predict_mixture_of_experts(
        model,
        predict_counts,
        predict_teachers,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    save_imputed_matrix(mu, output_root, "moe_fusion", dataset_id, disease, tissue)
    pi_path = Path(output_root) / "moe_fusion" / dataset_id / disease / f"{tissue}_pi.npy"
    np.save(pi_path, pi.astype(np.float32))
    masked_counts = predict_counts.copy()
    masked_counts[eval_mask] = 0.0
    if not np.allclose(masked_counts[eval_mask], 0.0):
        raise RuntimeError("Masked inputs still contain non-zero entries.")
    eval_teachers = build_eval_teachers(
        masked_counts,
        predict_teachers,
        list(predict_teachers.keys()),
        eval_mask,
        eval_strategy,
        eval_svd_components,
        eval_knn_neighbors,
        eval_graph_neighbors,
        eval_graph_pca_components,
        eval_graph_include_self,
        seed,
    )
    masked_mu, _ = predict_mixture_of_experts(
        model,
        masked_counts,
        eval_teachers,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )
    metrics = masked_reconstruction_metrics(
        predict_counts,
        masked_mu,
        mask_fraction=cfg.mask_fraction,
        seed=seed,
        mask=eval_mask,
    )
    output_dir = Path(output_root) / "moe_fusion" / dataset_id / disease
    save_metrics(output_dir, metrics, "masked_metrics.json")
    write_sanity_samples(
        output_dir,
        "masked",
        predict_counts,
        masked_counts,
        eval_teachers,
        masked_mu,
        eval_mask,
        eval_debug_samples,
        seed,
    )
    save_metadata(
        Path(output_root) / "moe_fusion" / dataset_id / disease,
        {"history": history, "config": cfg.__dict__, "subset": subset_info},
    )


def run_diffusion(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    cfg: TrainConfig,
    output_root: str,
    dataset_id: str,
    disease: str,
    tissue: str,
    seed: int,
    timesteps: int,
    subset_info: Dict[str, int],
    predict_counts: np.ndarray,
    predict_teachers: Dict[str, np.ndarray],
    eval_mask: np.ndarray,
    eval_strategy: str,
    eval_svd_components: int,
    eval_knn_neighbors: int,
    eval_graph_neighbors: int,
    eval_graph_pca_components: int,
    eval_graph_include_self: bool,
    eval_debug_samples: int,
):
    log("Training Diffusion-Guided Denoiser")
    model, history = train_diffusion(counts, teachers, cfg, seed=seed, timesteps=timesteps)
    mu = predict_diffusion(
        model,
        predict_counts,
        predict_teachers,
        batch_size=max(16, cfg.batch_size // 2),
        device=cfg.device,
    )
    save_imputed_matrix(mu, output_root, "diffusion_guided", dataset_id, disease, tissue)
    masked_counts = predict_counts.copy()
    masked_counts[eval_mask] = 0.0
    if not np.allclose(masked_counts[eval_mask], 0.0):
        raise RuntimeError("Masked inputs still contain non-zero entries.")
    eval_teachers = build_eval_teachers(
        masked_counts,
        predict_teachers,
        list(predict_teachers.keys()),
        eval_mask,
        eval_strategy,
        eval_svd_components,
        eval_knn_neighbors,
        eval_graph_neighbors,
        eval_graph_pca_components,
        eval_graph_include_self,
        seed,
    )
    masked_mu = predict_diffusion(
        model,
        masked_counts,
        eval_teachers,
        batch_size=max(16, cfg.batch_size // 2),
        device=cfg.device,
    )
    metrics = masked_reconstruction_metrics(
        predict_counts,
        masked_mu,
        mask_fraction=cfg.mask_fraction,
        seed=seed,
        mask=eval_mask,
    )
    output_dir = Path(output_root) / "diffusion_guided" / dataset_id / disease
    save_metrics(output_dir, metrics, "masked_metrics.json")
    write_sanity_samples(
        output_dir,
        "masked",
        predict_counts,
        masked_counts,
        eval_teachers,
        masked_mu,
        eval_mask,
        eval_debug_samples,
        seed,
    )
    save_metadata(
        Path(output_root) / "diffusion_guided" / dataset_id / disease,
        {"history": history, "config": cfg.__dict__, "timesteps": timesteps, "subset": subset_info},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion imputation models")
    parser.add_argument("--input_file", required=True, help="Path to h5ad file with counts")
    parser.add_argument("--disease", required=True, help="Disease label for output paths")
    parser.add_argument("--tissue", required=True, help="Tissue label for output paths")
    parser.add_argument("--output_root", default="output", help="Root directory for saved outputs")
    parser.add_argument(
        "--teacher_root",
        default="output",
        help="Root directory for cached teacher matrices.",
    )
    parser.add_argument("--dataset_id", default=None, help="Optional dataset id override")
    parser.add_argument("--teacher_methods", default="gene_median,svd_impute", help="Comma-separated teacher methods")
    parser.add_argument(
        "--teacher_likelihood_exclude",
        type=str,
        default="",
        help="Comma-separated teacher names to exclude from likelihood fusion.",
    )
    parser.add_argument("--max_cells", type=int, default=None, help="Optional cell subsample")
    parser.add_argument("--max_genes", type=int, default=None, help="Optional gene subsample")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_fraction", type=float, default=0.15)
    parser.add_argument("--teacher_weight", type=float, default=1.0)
    parser.add_argument("--best_teacher_weight", type=float, default=0.0)
    parser.add_argument("--best_teacher_min_log", type=float, default=0.0)
    parser.add_argument("--best_teacher_exclude", type=str, default="", help="Comma-separated teacher names to exclude from best-teacher selection.")
    parser.add_argument("--best_teacher_temp", type=float, default=0.5)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--latent_prior_scale", type=float, default=1.0)
    parser.add_argument("--teacher_warmup_epochs", type=int, default=1)
    parser.add_argument("--teacher_ramp_epochs", type=int, default=3)
    parser.add_argument("--teacher_dropout", type=float, default=0.0)
    parser.add_argument("--teacher_loss_on_post", action="store_true")
    parser.add_argument(
        "--teacher_calibration_weight",
        type=float,
        default=1e-3,
        help="Regularization weight for per-teacher affine calibration.",
    )
    parser.add_argument(
        "--latent_use_pca",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include PCA cell embeddings in sigma-net features.",
    )
    parser.add_argument("--latent_pca_components", type=int, default=30)
    parser.add_argument("--latent_pca_proj_dim", type=int, default=8)
    parser.add_argument(
        "--eval_teacher_strategy",
        choices=["auto", "mask", "recompute"],
        default="auto",
        help="How to prepare teachers for masked evaluation.",
    )
    parser.add_argument("--eval_svd_components", type=int, default=50)
    parser.add_argument("--eval_knn_neighbors", type=int, default=5)
    parser.add_argument("--graph_neighbors", type=int, default=30)
    parser.add_argument("--graph_pca_components", type=int, default=30)
    parser.add_argument("--graph_include_self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_debug_samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument(
        "--methods",
        default="latent_truth,moe_fusion,diffusion_guided",
        help="Comma-separated list of fusion methods to run.",
    )
    args = parser.parse_args()

    dataset_id = args.dataset_id or infer_dataset_id(args.input_file)
    counts, _, _, _, _ = load_counts(
        args.input_file,
        max_cells=None,
        max_genes=None,
        seed=args.seed,
    )
    log(f"Loaded full counts with shape {counts.shape}")

    methods = [m.strip() for m in args.teacher_methods.split(",") if m.strip()]
    teachers = load_teacher_matrices(
        methods=methods,
        output_root=args.teacher_root,
        dataset_id=dataset_id,
        disease=args.disease,
        tissue=args.tissue,
        shape=counts.shape,
        allow_compute=True,
        counts=counts,
        graph_neighbors=args.graph_neighbors,
        graph_pca_components=args.graph_pca_components,
        graph_include_self=args.graph_include_self,
        seed=args.seed,
    )
    log(f"Loaded teachers: {list(teachers.keys())}")

    rng = np.random.default_rng(args.seed)
    cell_idx = np.arange(counts.shape[0])
    gene_idx = np.arange(counts.shape[1])
    if args.max_cells and args.max_cells < counts.shape[0]:
        cell_idx = rng.choice(counts.shape[0], size=args.max_cells, replace=False)
    if args.max_genes and args.max_genes < counts.shape[1]:
        gene_idx = rng.choice(counts.shape[1], size=args.max_genes, replace=False)
    train_counts = counts[cell_idx][:, gene_idx]
    train_teachers = {name: mat[cell_idx][:, gene_idx] for name, mat in teachers.items()}
    if train_counts.shape[1] != counts.shape[1]:
        log("Gene subsampling active: predictions will only cover the sampled genes.")
        predict_counts = train_counts
        predict_teachers = train_teachers
    else:
        predict_counts = counts
        predict_teachers = teachers
    gene_mean = np.log1p(np.mean(counts, axis=0)).astype(np.float32)
    gene_dropout = np.mean(counts <= 0, axis=0).astype(np.float32)
    if predict_counts.shape[1] != counts.shape[1]:
        gene_mean = gene_mean[gene_idx]
        gene_dropout = gene_dropout[gene_idx]
    predict_loglib = np.log1p(np.sum(predict_counts, axis=1)).astype(np.float32)
    eval_mask = sample_mask(predict_counts, mask_fraction=args.mask_fraction, seed=args.seed)
    if eval_mask.sum() == 0:
        raise RuntimeError("Evaluation mask produced no entries.")

    cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        mask_fraction=args.mask_fraction,
        teacher_weight=args.teacher_weight,
        best_teacher_weight=args.best_teacher_weight,
        best_teacher_min_log=args.best_teacher_min_log,
        best_teacher_exclude=tuple(name.strip() for name in args.best_teacher_exclude.split(",") if name.strip()),
        best_teacher_temp=args.best_teacher_temp,
        entropy_weight=args.entropy_weight,
        prior_scale=args.latent_prior_scale,
        teacher_warmup_epochs=args.teacher_warmup_epochs,
        teacher_ramp_epochs=args.teacher_ramp_epochs,
        teacher_dropout=args.teacher_dropout,
        teacher_loss_on_prior=not args.teacher_loss_on_post,
        teacher_calibration_weight=args.teacher_calibration_weight,
        teacher_likelihood_exclude=tuple(
            name.strip() for name in args.teacher_likelihood_exclude.split(",") if name.strip()
        ),
        device=args.device,
    )
    subset_info = {
        "train_cells": int(train_counts.shape[0]),
        "train_genes": int(train_counts.shape[1]),
        "full_cells": int(counts.shape[0]),
        "full_genes": int(counts.shape[1]),
        "max_cells": int(args.max_cells) if args.max_cells else None,
        "max_genes": int(args.max_genes) if args.max_genes else None,
    }

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    pca_model = None
    train_pca = None
    predict_pca = None
    pca_components = 0
    if args.latent_use_pca:
        pca_model, train_pca, predict_pca = build_pca_features(
            train_counts,
            predict_counts,
            mask_fraction=args.mask_fraction,
            n_components=args.latent_pca_components,
            seed=args.seed,
        )
        pca_components = args.latent_pca_components
        log(f"Computed PCA features with {args.latent_pca_components} components")

    if "latent_truth" in requested:
        run_latent_truth(
            train_counts,
            train_teachers,
            cfg,
            args.output_root,
            dataset_id,
            args.disease,
            args.tissue,
            args.seed,
            subset_info,
            predict_counts,
            predict_teachers,
            eval_mask,
            args.eval_teacher_strategy,
            args.eval_svd_components,
            args.eval_knn_neighbors,
            args.graph_neighbors,
            args.graph_pca_components,
            args.graph_include_self,
            args.eval_debug_samples,
            predict_loglib,
            gene_mean,
            gene_dropout,
            pca_model,
            train_pca,
            predict_pca,
            pca_components,
            args.latent_pca_proj_dim,
        )
    if "moe_fusion" in requested:
        run_mixture_of_experts(
            train_counts,
            train_teachers,
            cfg,
            args.output_root,
            dataset_id,
            args.disease,
            args.tissue,
            args.seed,
            subset_info,
            predict_counts,
            predict_teachers,
            eval_mask,
            args.eval_teacher_strategy,
            args.eval_svd_components,
            args.eval_knn_neighbors,
            args.graph_neighbors,
            args.graph_pca_components,
            args.graph_include_self,
            args.eval_debug_samples,
        )
    if "diffusion_guided" in requested:
        run_diffusion(
            train_counts,
            train_teachers,
            cfg,
            args.output_root,
            dataset_id,
            args.disease,
            args.tissue,
            args.seed,
            timesteps=args.diffusion_steps,
            subset_info=subset_info,
            predict_counts=predict_counts,
            predict_teachers=predict_teachers,
            eval_mask=eval_mask,
            eval_strategy=args.eval_teacher_strategy,
            eval_svd_components=args.eval_svd_components,
            eval_knn_neighbors=args.eval_knn_neighbors,
            eval_graph_neighbors=args.graph_neighbors,
            eval_graph_pca_components=args.graph_pca_components,
            eval_graph_include_self=args.graph_include_self,
            eval_debug_samples=args.eval_debug_samples,
        )


if __name__ == "__main__":
    main()
