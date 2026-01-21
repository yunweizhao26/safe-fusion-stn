from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from fusion.models import DiffusionGuidedModel, LatentTruthModel, MixtureOfExpertsModel


def _batch_indices(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        yield slice(start, end)


def predict_latent_truth(
    model: LatentTruthModel,
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray] = None,
    fuse: bool = False,
    cell_loglib: np.ndarray = None,
    gene_mean: np.ndarray = None,
    gene_dropout: np.ndarray = None,
    pca_features: np.ndarray = None,
    teacher_mask: np.ndarray = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    teacher_stack = None
    if teachers:
        teacher_stack = np.stack([teachers[name] for name in teachers.keys()], axis=0)
    mu_list, pi_list = [], []
    var_list = []
    with torch.no_grad():
        for sl in _batch_indices(counts.shape[0], batch_size):
            x = torch.from_numpy(counts[sl]).to(device)
            x_log = torch.log1p(x)
            mu, pi = model(x_log)
            var_post = None
            if fuse and teacher_stack is not None:
                t = torch.from_numpy(teacher_stack[:, sl, :]).to(device)
                if cell_loglib is None or gene_mean is None or gene_dropout is None:
                    raise ValueError("cell_loglib, gene_mean, gene_dropout required for fusion")
                cell_feat = torch.from_numpy(cell_loglib[sl]).to(device)
                gene_mean_t = torch.from_numpy(gene_mean).to(device)
                gene_dropout_t = torch.from_numpy(gene_dropout).to(device)
                pca_batch = None
                if pca_features is not None:
                    pca_batch = torch.from_numpy(pca_features[sl]).to(device)
                mask_t = None
                if teacher_mask is not None:
                    mask_t = torch.from_numpy(teacher_mask.astype(np.float32)).to(device)
                mu, var_post = model.fuse_mu(
                    mu,
                    t,
                    cell_feat,
                    gene_mean_t,
                    gene_dropout_t,
                    pca_batch,
                    teacher_mask=mask_t,
                )
            mu_list.append(mu.cpu().numpy())
            pi_list.append(pi.cpu().numpy())
            if var_post is not None:
                var_list.append(var_post.cpu().numpy())
    mu_out = np.vstack(mu_list)
    pi_out = np.vstack(pi_list)
    if var_list:
        var_out = np.vstack(var_list)
    else:
        var_out = np.zeros_like(mu_out)
    return mu_out, pi_out, var_out


def predict_mixture_of_experts(
    model: MixtureOfExpertsModel,
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    teacher_stack = np.stack([teachers[name] for name in teachers.keys()], axis=0)
    mu_list, pi_list = [], []
    with torch.no_grad():
        for sl in _batch_indices(counts.shape[0], batch_size):
            x = torch.from_numpy(counts[sl]).to(device)
            t = torch.from_numpy(teacher_stack[:, sl, :]).to(device)
            x_log = torch.log1p(x)
            mu, pi, _ = model(x_log, t)
            mu_list.append(mu.cpu().numpy())
            pi_list.append(pi.cpu().numpy())
    return np.vstack(mu_list), np.vstack(pi_list)


def predict_diffusion(
    model: DiffusionGuidedModel,
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    teacher_stack = np.stack([teachers[name] for name in teachers.keys()], axis=0)
    mu_list = []
    with torch.no_grad():
        for sl in _batch_indices(counts.shape[0], batch_size):
            x = torch.from_numpy(counts[sl]).to(device)
            t = torch.from_numpy(teacher_stack[:, sl, :]).to(device)
            x_log = torch.log1p(x)
            teacher_mean = torch.log1p(torch.clamp(t, min=0.0)).mean(dim=0)
            denoised_log = model.denoise(x_log, teacher_mean)
            mu = torch.expm1(denoised_log).clamp(min=0.0)
            mu_list.append(mu.cpu().numpy())
    return np.vstack(mu_list)


def sample_mask(
    counts: np.ndarray,
    mask_fraction: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nonzero = counts > 0
    return (rng.random(counts.shape) < mask_fraction) & nonzero


def masked_reconstruction_metrics(
    counts: np.ndarray,
    prediction: np.ndarray,
    mask_fraction: float = 0.15,
    seed: int = 42,
    mask: np.ndarray = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    nonzero = counts > 0
    if mask is None:
        mask = (rng.random(counts.shape) < mask_fraction) & nonzero
    if mask.sum() == 0:
        raise ValueError("Mask produced no entries to evaluate.")
    true_vals = counts[mask]
    pred_vals = prediction[mask]

    log_true = np.log1p(true_vals)
    log_pred = np.log1p(np.clip(pred_vals, a_min=0.0, a_max=None))
    rmse = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    mae = float(np.mean(np.abs(log_true - log_pred)))

    zero_mask = ~nonzero
    zero_idx = np.column_stack(np.where(zero_mask))
    sample_size = min(len(true_vals), zero_idx.shape[0])
    sample_idx = rng.choice(zero_idx.shape[0], size=sample_size, replace=False)
    sampled = zero_idx[sample_idx]
    zero_scores = prediction[sampled[:, 0], sampled[:, 1]]

    labels = np.concatenate([np.ones(len(true_vals)), np.zeros(sample_size)])
    scores = np.concatenate([pred_vals, zero_scores])
    ap = float(average_precision_score(labels, scores))
    try:
        roc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc = 0.5
    return {
        "rmse_log1p": rmse,
        "mae_log1p": mae,
        "pr_auc": ap,
        "roc_auc": roc,
    }
