from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fusion.models import (
    DiffusionGuidedModel,
    LatentTruthModel,
    MixtureOfExpertsModel,
    zinb_negative_log_likelihood,
)


class IndexDataset(Dataset):
    """Simple dataset that yields indices for batching."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    mask_fraction: float = 0.15
    teacher_weight: float = 1.0
    entropy_weight: float = 0.1
    prior_scale: float = 1.0
    teacher_warmup_epochs: int = 1
    teacher_ramp_epochs: int = 3
    teacher_dropout: float = 0.0
    teacher_loss_on_prior: bool = True
    best_teacher_weight: float = 0.0
    best_teacher_min_log: float = 0.0
    best_teacher_exclude: tuple = ()
    best_teacher_temp: float = 0.5
    teacher_calibration_weight: float = 1e-3
    teacher_likelihood_exclude: tuple = ()
    device: str = "cpu"


def split_indices(n_cells: int, seed: int, val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_cells)
    rng.shuffle(idx)
    split = int(n_cells * (1.0 - val_fraction))
    return idx[:split], idx[split:]


def build_dataloaders(
    n_cells: int,
    seed: int,
    batch_size: int,
    val_fraction: float = 0.1,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_idx, val_idx = split_indices(n_cells, seed, val_fraction)
    train_loader = DataLoader(IndexDataset(len(train_idx)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(IndexDataset(len(val_idx)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_idx, val_idx


def scheduled_weight(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    max_weight: float,
) -> float:
    if max_weight <= 0.0:
        return 0.0
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs > 0:
        progress = min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
        return max_weight * progress
    return max_weight


def _get_batch(counts: np.ndarray, teachers: np.ndarray, idx_map: np.ndarray, batch_idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    cell_idx = idx_map[batch_idx]
    x = torch.from_numpy(counts[cell_idx])
    t = torch.from_numpy(teachers[:, cell_idx, :])
    return x, t


def _mask_nonzero(x: torch.Tensor, mask_fraction: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = (x > 0) & (torch.rand_like(x) < mask_fraction)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask.float()


def train_latent_truth(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    config: TrainConfig,
    gene_mean: np.ndarray,
    gene_dropout: np.ndarray,
    pca_features: np.ndarray = None,
    pca_proj_dim: int = 8,
    seed: int = 42,
) -> Tuple[LatentTruthModel, Dict[str, float]]:
    teacher_names = list(teachers.keys())
    teacher_stack = np.stack([teachers[name] for name in teacher_names], axis=0)
    train_loader, val_loader, train_idx, val_idx = build_dataloaders(
        counts.shape[0], seed, config.batch_size
    )
    pca_dim = int(pca_features.shape[1]) if pca_features is not None else 0
    model = LatentTruthModel(
        n_genes=counts.shape[1],
        n_teachers=teacher_stack.shape[0],
        pca_dim=pca_dim,
        pca_proj_dim=pca_proj_dim,
    ).to(config.device)
    pca_feat_t = None
    if pca_features is not None:
        pca_feat_t = torch.from_numpy(pca_features).to(config.device)
    gene_mean_t = torch.from_numpy(gene_mean).to(config.device)
    gene_dropout_t = torch.from_numpy(gene_dropout).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = {"train_loss": [], "val_loss": []}
    selection_idx = [i for i, name in enumerate(teacher_names) if name not in config.best_teacher_exclude]
    selection_idx_t = (
        torch.tensor(selection_idx, device=config.device, dtype=torch.long)
        if len(selection_idx) > 0
        else None
    )
    likelihood_mask = None
    if config.teacher_likelihood_exclude:
        likelihood_mask = torch.ones(len(teacher_names), device=config.device)
        for idx, name in enumerate(teacher_names):
            if name in config.teacher_likelihood_exclude:
                likelihood_mask[idx] = 0.0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        teacher_lambda = scheduled_weight(
            epoch,
            config.teacher_warmup_epochs,
            config.teacher_ramp_epochs,
            config.teacher_weight,
        )
        best_lambda = scheduled_weight(
            epoch,
            config.teacher_warmup_epochs,
            config.teacher_ramp_epochs,
            config.best_teacher_weight,
        )
        for batch_idx in train_loader:
            x, t = _get_batch(counts, teacher_stack, train_idx, batch_idx.numpy())
            x = x.to(config.device)
            t = t.to(config.device)
            x_masked, mask = _mask_nonzero(x, config.mask_fraction)
            x_log = torch.log1p(x_masked)
            mu, pi = model(x_log)
            mu_log = torch.log1p(mu)
            teacher_log = torch.log1p(torch.clamp(t, min=0.0))
            teacher_log = model.calibrate_teacher_log(teacher_log)
            cell_feat = torch.log1p(torch.sum(x_masked, dim=1))
            pca_batch = pca_feat_t[train_idx[batch_idx.numpy()]] if pca_feat_t is not None else None
            var, bias = model.predict_teacher_params(
                teacher_log,
                mu_log,
                cell_feat,
                gene_mean_t,
                gene_dropout_t,
                pca_batch,
            )
            drop_mask = None
            if config.teacher_dropout > 0:
                keep = (torch.rand(var.shape[0], device=config.device) > config.teacher_dropout).float()
                if torch.sum(keep) > 0:
                    drop_mask = keep
            teacher_mask = None
            if likelihood_mask is not None:
                teacher_mask = likelihood_mask
            if drop_mask is not None:
                teacher_mask = drop_mask if teacher_mask is None else teacher_mask * drop_mask
            mu_post_log, _ = model.posterior_fuse(mu_log, teacher_log, var, bias, teacher_mask=teacher_mask)
            mu_post = torch.expm1(mu_post_log).clamp(min=0.0)
            theta = model.dispersion()
            zinb = zinb_negative_log_likelihood(x, mu_post, theta, pi, mask)
            teacher_center = mu_log if config.teacher_loss_on_prior else mu_post_log
            teacher_nll = model.teacher_nll(teacher_center, teacher_log, var, bias, mask, teacher_mask=teacher_mask)
            best_loss = torch.tensor(0.0, device=config.device)
            cal_loss = torch.tensor(0.0, device=config.device)
            if best_lambda > 0.0 and selection_idx_t is not None:
                x_log_true = torch.log1p(torch.clamp(x, min=0.0))
                distill_mask = mask.bool() & (x_log_true > config.best_teacher_min_log)
                teacher_log_select = teacher_log.index_select(0, selection_idx_t)
                errors = torch.abs(teacher_log_select - x_log_true.unsqueeze(0))
                if drop_mask is not None:
                    keep = drop_mask.index_select(0, selection_idx_t).view(-1, 1, 1)
                    errors = torch.where(keep > 0, errors, torch.full_like(errors, 1e6))
                temp = max(config.best_teacher_temp, 1e-3)
                weights = torch.softmax(-errors / temp, dim=0)
                diff = mu_post_log.unsqueeze(0) - teacher_log_select
                weighted_sq = torch.sum(weights * (diff ** 2), dim=0)
                if distill_mask.any():
                    best_loss = torch.sum(weighted_sq * distill_mask) / (distill_mask.sum() + 1e-8)
            if config.teacher_calibration_weight > 0.0:
                cal_loss = model.teacher_calibration_loss()
            loss = zinb + teacher_lambda * teacher_nll + best_lambda * best_loss + config.teacher_calibration_weight * cal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            teacher_lambda = scheduled_weight(
                epoch,
                config.teacher_warmup_epochs,
                config.teacher_ramp_epochs,
                config.teacher_weight,
            )
            best_lambda = scheduled_weight(
                epoch,
                config.teacher_warmup_epochs,
                config.teacher_ramp_epochs,
                config.best_teacher_weight,
            )
            for batch_idx in val_loader:
                x, t = _get_batch(counts, teacher_stack, val_idx, batch_idx.numpy())
                x = x.to(config.device)
                t = t.to(config.device)
                x_masked, mask = _mask_nonzero(x, config.mask_fraction)
                x_log = torch.log1p(x_masked)
                mu, pi = model(x_log)
                mu_log = torch.log1p(mu)
                teacher_log = torch.log1p(torch.clamp(t, min=0.0))
                teacher_log = model.calibrate_teacher_log(teacher_log)
                cell_feat = torch.log1p(torch.sum(x_masked, dim=1))
                pca_batch = pca_feat_t[val_idx[batch_idx.numpy()]] if pca_feat_t is not None else None
                var, bias = model.predict_teacher_params(
                    teacher_log,
                    mu_log,
                    cell_feat,
                    gene_mean_t,
                    gene_dropout_t,
                    pca_batch,
                )
                teacher_mask = likelihood_mask
                mu_post_log, _ = model.posterior_fuse(mu_log, teacher_log, var, bias, teacher_mask=teacher_mask)
                mu_post = torch.expm1(mu_post_log).clamp(min=0.0)
                theta = model.dispersion()
                zinb = zinb_negative_log_likelihood(x, mu_post, theta, pi, mask)
                teacher_center = mu_log if config.teacher_loss_on_prior else mu_post_log
                teacher_nll = model.teacher_nll(teacher_center, teacher_log, var, bias, mask, teacher_mask=teacher_mask)
                best_loss = torch.tensor(0.0, device=config.device)
                cal_loss = torch.tensor(0.0, device=config.device)
                if best_lambda > 0.0 and selection_idx_t is not None:
                    x_log_true = torch.log1p(torch.clamp(x, min=0.0))
                    distill_mask = mask.bool() & (x_log_true > config.best_teacher_min_log)
                    teacher_log_select = teacher_log.index_select(0, selection_idx_t)
                    errors = torch.abs(teacher_log_select - x_log_true.unsqueeze(0))
                    temp = max(config.best_teacher_temp, 1e-3)
                    weights = torch.softmax(-errors / temp, dim=0)
                    diff = mu_post_log.unsqueeze(0) - teacher_log_select
                    weighted_sq = torch.sum(weights * (diff ** 2), dim=0)
                    if distill_mask.any():
                        best_loss = torch.sum(weighted_sq * distill_mask) / (distill_mask.sum() + 1e-8)
                if config.teacher_calibration_weight > 0.0:
                    cal_loss = model.teacher_calibration_loss()
                loss = zinb + teacher_lambda * teacher_nll + best_lambda * best_loss + config.teacher_calibration_weight * cal_loss
                val_losses.append(float(loss.detach().cpu()))
        history["val_loss"].append(float(np.mean(val_losses)))
    return model, history


def train_mixture_of_experts(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    config: TrainConfig,
    seed: int = 42,
) -> Tuple[MixtureOfExpertsModel, Dict[str, float]]:
    teacher_stack = np.stack([teachers[name] for name in teachers.keys()], axis=0)
    train_loader, val_loader, train_idx, val_idx = build_dataloaders(
        counts.shape[0], seed, config.batch_size
    )
    model = MixtureOfExpertsModel(
        n_teachers=teacher_stack.shape[0],
        n_genes=counts.shape[1],
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch_idx in train_loader:
            x, t = _get_batch(counts, teacher_stack, train_idx, batch_idx.numpy())
            x = x.to(config.device)
            t = t.to(config.device)
            x_masked, mask = _mask_nonzero(x, config.mask_fraction)
            x_log = torch.log1p(x_masked)
            mu, pi, alpha = model(x_log, t)
            theta = model.dispersion()
            zinb = zinb_negative_log_likelihood(x, mu, theta, pi, mask)

            # Agreement-aware entropy penalty: higher disagreement -> lower entropy
            teacher_log = torch.log1p(torch.clamp(t, min=0.0))
            disagreement = torch.std(teacher_log, dim=0)
            entropy = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=0)
            entropy_penalty = torch.mean(disagreement * entropy)

            loss = zinb + config.entropy_weight * entropy_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx in val_loader:
                x, t = _get_batch(counts, teacher_stack, val_idx, batch_idx.numpy())
                x = x.to(config.device)
                t = t.to(config.device)
                x_masked, mask = _mask_nonzero(x, config.mask_fraction)
                x_log = torch.log1p(x_masked)
                mu, pi, alpha = model(x_log, t)
                theta = model.dispersion()
                zinb = zinb_negative_log_likelihood(x, mu, theta, pi, mask)
                teacher_log = torch.log1p(torch.clamp(t, min=0.0))
                disagreement = torch.std(teacher_log, dim=0)
                entropy = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=0)
                entropy_penalty = torch.mean(disagreement * entropy)
                loss = zinb + config.entropy_weight * entropy_penalty
                val_losses.append(float(loss.detach().cpu()))
        history["val_loss"].append(float(np.mean(val_losses)))
    return model, history


def train_diffusion(
    counts: np.ndarray,
    teachers: Dict[str, np.ndarray],
    config: TrainConfig,
    seed: int = 42,
    timesteps: int = 20,
) -> Tuple[DiffusionGuidedModel, Dict[str, float]]:
    teacher_stack = np.stack([teachers[name] for name in teachers.keys()], axis=0)
    train_loader, val_loader, train_idx, val_idx = build_dataloaders(
        counts.shape[0], seed, config.batch_size
    )
    model = DiffusionGuidedModel(
        n_genes=counts.shape[1],
        timesteps=timesteps,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = {"train_loss": [], "val_loss": []}
    rng = np.random.default_rng(seed)

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch_idx in train_loader:
            x, t = _get_batch(counts, teacher_stack, train_idx, batch_idx.numpy())
            x = x.to(config.device)
            t = t.to(config.device)
            x_log = torch.log1p(x)
            teacher_mean = torch.log1p(torch.clamp(t, min=0.0)).mean(dim=0)
            noise = torch.randn_like(x_log)
            t_idx = torch.from_numpy(rng.integers(0, model.timesteps, size=x_log.shape[0])).to(config.device)
            x_t = model.q_sample(x_log, t_idx, noise)
            eps_pred = model(x_t, t_idx, teacher_mean)
            loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx in val_loader:
                x, t = _get_batch(counts, teacher_stack, val_idx, batch_idx.numpy())
                x = x.to(config.device)
                t = t.to(config.device)
                x_log = torch.log1p(x)
                teacher_mean = torch.log1p(torch.clamp(t, min=0.0)).mean(dim=0)
                noise = torch.randn_like(x_log)
                t_idx = torch.from_numpy(rng.integers(0, model.timesteps, size=x_log.shape[0])).to(config.device)
                x_t = model.q_sample(x_log, t_idx, noise)
                eps_pred = model(x_t, t_idx, teacher_mean)
                loss = F.mse_loss(eps_pred, noise)
                val_losses.append(float(loss.detach().cpu()))
        history["val_loss"].append(float(np.mean(val_losses)))
    return model, history
