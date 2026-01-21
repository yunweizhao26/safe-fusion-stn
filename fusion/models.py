from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def zinb_negative_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    ZINB log-likelihood per entry:
        NB(x; mu, theta) = Gamma(x + theta) / (Gamma(theta) * x!)
                           * (theta / (theta + mu))^theta
                           * (mu / (theta + mu))^x

        ZINB(x; pi, mu, theta) = pi + (1 - pi) * NB(x)   if x == 0
                               (1 - pi) * NB(x)        otherwise
    """
    eps = 1e-8
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)
    pi = torch.clamp(pi, min=eps, max=1.0 - eps)

    log_theta_mu = torch.log(theta + mu)
    log_nb = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )

    zero_mask = (x <= 0.0).float()
    log_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb) + eps)
    log_nonzero = torch.log(1.0 - pi + eps) + log_nb
    log_likelihood = zero_mask * log_zero + (1.0 - zero_mask) * log_nonzero
    nll = -log_likelihood
    return (nll * mask).sum() / (mask.sum() + eps)


class LatentTruthModel(nn.Module):
    """
    Latent Truth Fusion model.

    Teacher likelihood (log scale):
        y_m = log(x_hat_m + eps)
        y_m ~ Normal(mu_log + b_m, var_m(g,c))
    """

    def __init__(
        self,
        n_genes: int,
        n_teachers: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        sigma_feature_dim: int = None,
        sigma_hidden_dim: int = 64,
        use_bias: bool = True,
        var_min: float = 1e-3,
        var_max: float = 2.0,
        prior_var_min: float = 1e-3,
        prior_var_max: float = 2.0,
        pca_dim: int = 0,
        pca_proj_dim: int = 8,
        normalize_sigma_features: bool = True,
        detach_mu_for_sigma: bool = True,
    ):
        super().__init__()
        init_scale = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )
        self.decoder_pi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )
        # Per-gene dispersion (theta > 0)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        # Prior variance per gene
        self.prior_logvar = nn.Parameter(torch.zeros(n_genes))
        self.prior_var_min = prior_var_min
        self.prior_var_max = prior_var_max

        self.n_teachers = n_teachers
        self.teacher_affine_eps = 1e-3
        self.teacher_affine_raw_scale = nn.Parameter(torch.full((n_teachers,), init_scale))
        self.teacher_affine_bias = nn.Parameter(torch.zeros(n_teachers))
        self.use_bias = use_bias
        self.var_min = var_min
        self.var_max = var_max
        self.normalize_sigma_features = normalize_sigma_features
        self.detach_mu_for_sigma = detach_mu_for_sigma

        self.pca_proj = None
        self.pca_proj_dim = 0
        if pca_dim and pca_dim > 0:
            self.pca_proj = nn.Sequential(
                nn.Linear(pca_dim, pca_proj_dim),
                nn.ReLU(),
            )
            self.pca_proj_dim = pca_proj_dim

        output_mult = 2 if use_bias else 1
        base_feature_dim = 7
        if sigma_feature_dim is None:
            sigma_feature_dim = base_feature_dim + self.pca_proj_dim
        self.sigma_net = nn.Sequential(
            nn.Linear(sigma_feature_dim, sigma_hidden_dim),
            nn.ReLU(),
            nn.Linear(sigma_hidden_dim, n_teachers * output_mult),
        )

    def teacher_affine_scale(self) -> torch.Tensor:
        return F.softplus(self.teacher_affine_raw_scale) + self.teacher_affine_eps

    def calibrate_teacher_log(self, teacher_log: torch.Tensor) -> torch.Tensor:
        scale = self.teacher_affine_scale().view(-1, 1, 1)
        bias = self.teacher_affine_bias.view(-1, 1, 1)
        return scale * teacher_log + bias

    def teacher_calibration_loss(self) -> torch.Tensor:
        scale = self.teacher_affine_scale()
        return torch.sum((scale - 1.0) ** 2 + self.teacher_affine_bias**2)

    def forward(self, x_log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_log)
        mu = F.softplus(self.decoder_mu(z))
        pi = torch.sigmoid(self.decoder_pi(z))
        return mu, pi

    def _zscore(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)

    def build_sigma_features(
        self,
        teacher_log: torch.Tensor,
        mu_log: torch.Tensor,
        cell_loglib: torch.Tensor,
        gene_mean: torch.Tensor,
        gene_dropout: torch.Tensor,
        pca_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Construct sigma-net features per entry.
        Features: log_libsize, gene_mean, gene_dropout, mu_log, teacher_mean_log, teacher_std_log, teacher_range_log
        """
        mean_log = torch.mean(teacher_log, dim=0)
        std_log = torch.std(teacher_log, dim=0)
        min_log = torch.min(teacher_log, dim=0).values
        max_log = torch.max(teacher_log, dim=0).values
        range_log = max_log - min_log
        cell_feat = cell_loglib[:, None].expand(mean_log.shape[0], mean_log.shape[1])
        gene_mean_feat = gene_mean[None, :].expand(mean_log.shape[0], mean_log.shape[1])
        gene_dropout_feat = gene_dropout[None, :].expand(mean_log.shape[0], mean_log.shape[1])
        if self.normalize_sigma_features:
            cell_feat = self._zscore(cell_feat)
            gene_mean_feat = self._zscore(gene_mean_feat)
            gene_dropout_feat = self._zscore(gene_dropout_feat)
            mu_log = self._zscore(mu_log)
            mean_log = self._zscore(mean_log)
            std_log = self._zscore(std_log)
            range_log = self._zscore(range_log)
        features = torch.stack(
            [cell_feat, gene_mean_feat, gene_dropout_feat, mu_log, mean_log, std_log, range_log],
            dim=-1,
        )
        if pca_feat is not None and self.pca_proj is not None:
            pca_proj = self.pca_proj(pca_feat)
            pca_proj = pca_proj[:, None, :].expand(mean_log.shape[0], mean_log.shape[1], pca_proj.shape[1])
            features = torch.cat([features, pca_proj], dim=-1)
        return features

    def predict_teacher_params(
        self,
        teacher_log: torch.Tensor,
        mu_log: torch.Tensor,
        cell_loglib: torch.Tensor,
        gene_mean: torch.Tensor,
        gene_dropout: torch.Tensor,
        pca_feat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.detach_mu_for_sigma:
            mu_log = mu_log.detach()
        features = self.build_sigma_features(teacher_log, mu_log, cell_loglib, gene_mean, gene_dropout, pca_feat)
        flat = features.reshape(-1, features.shape[-1])
        raw = self.sigma_net(flat).view(
            features.shape[0],
            features.shape[1],
            self.n_teachers,
            -1,
        )
        raw_var = raw[..., 0]
        bias = raw[..., 1] if self.use_bias else torch.zeros_like(raw_var)
        var = self.var_min + (self.var_max - self.var_min) * torch.sigmoid(raw_var)
        var = var.permute(2, 0, 1)
        bias = bias.permute(2, 0, 1)
        return var, bias

    def posterior_fuse(
        self,
        mu_log: torch.Tensor,
        teacher_log: torch.Tensor,
        var: torch.Tensor,
        bias: torch.Tensor,
        teacher_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        prior_var = self.prior_var_min + (self.prior_var_max - self.prior_var_min) * torch.sigmoid(self.prior_logvar)
        prior_var = prior_var[None, :].expand(mu_log.shape[0], mu_log.shape[1])
        prior_prec = 1.0 / (prior_var + eps)
        weight = 1.0 / (var + eps)
        if teacher_mask is not None:
            weight = weight * teacher_mask[:, None, None]
        weighted_teacher = weight * (teacher_log - bias)
        num = mu_log * prior_prec + torch.sum(weighted_teacher, dim=0)
        denom = prior_prec + torch.sum(weight, dim=0)
        mu_post_log = num / (denom + eps)
        var_post = 1.0 / (denom + eps)
        return mu_post_log, var_post

    def teacher_nll(
        self,
        mu_post_log: torch.Tensor,
        teacher_log: torch.Tensor,
        var: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        teacher_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        eps = 1e-8
        residual = teacher_log - (mu_post_log[None, :, :] + bias)
        nll = 0.5 * (residual**2 / (var + eps) + torch.log(var + eps))
        if teacher_mask is not None:
            nll = nll * teacher_mask[:, None, None]
        mask = mask[None, :, :]
        return (nll * mask).sum() / (mask.sum() * nll.shape[0] + eps)

    def dispersion(self) -> torch.Tensor:
        return F.softplus(self.log_theta)

    def fuse_mu(
        self,
        mu: torch.Tensor,
        teachers: torch.Tensor,
        cell_loglib: torch.Tensor,
        gene_mean: torch.Tensor,
        gene_dropout: torch.Tensor,
        pca_feat: torch.Tensor = None,
        teacher_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Posterior fusion in log-space:
            mu_post = (p0*mu0 + sum_m pm*(y_m - b_m)) / (p0 + sum_m pm)
            var_post = 1 / (p0 + sum_m pm)
        """
        mu_log = torch.log1p(mu)
        teacher_log = torch.log1p(torch.clamp(teachers, min=0.0))
        teacher_log = self.calibrate_teacher_log(teacher_log)
        var, bias = self.predict_teacher_params(teacher_log, mu_log, cell_loglib, gene_mean, gene_dropout, pca_feat)
        mu_post_log, var_post = self.posterior_fuse(mu_log, teacher_log, var, bias, teacher_mask=teacher_mask)
        mu_post = torch.expm1(mu_post_log).clamp(min=0.0)
        return mu_post, var_post


class MixtureOfExpertsModel(nn.Module):
    """
    Mixture-of-experts gating model.

    Prediction:
        alpha_m = softmax(h(features))
        mu = sum_m alpha_m * x_hat_m
    """

    def __init__(
        self,
        n_teachers: int,
        n_genes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_teachers = n_teachers
        feature_dim = n_teachers + 2
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_teachers),
        )
        self.pi_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        x_log: torch.Tensor,
        teachers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-entry gating.
        Inputs:
            x_log: (B, G) log1p counts
            teachers: (M, B, G) raw teacher counts
        Returns:
            mu: (B, G)
            pi: (B, G)
            alpha: (M, B, G)
        """
        teacher_log = torch.log1p(torch.clamp(teachers, min=0.0))
        teacher_std = torch.std(teacher_log, dim=0)
        feats = torch.cat(
            [
                x_log[None, :, :],
                teacher_log,
                teacher_std[None, :, :],
            ],
            dim=0,
        )
        feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])
        logits = self.gate(feats)
        alpha = torch.softmax(logits, dim=-1)
        pi = torch.sigmoid(self.pi_head(feats))
        alpha = alpha.view(x_log.shape[0], x_log.shape[1], self.n_teachers).permute(2, 0, 1)
        mu = torch.sum(alpha * teachers, dim=0)
        pi = pi.view(x_log.shape[0], x_log.shape[1])
        return mu, pi, alpha

    def dispersion(self) -> torch.Tensor:
        return F.softplus(self.log_theta)


class DiffusionGuidedModel(nn.Module):
    """
    Lightweight diffusion denoiser with teacher guidance.

    Forward noise process:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Train to predict eps with teacher conditioning.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 256,
        timesteps: int = 20,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.net = nn.Sequential(
            nn.Linear(n_genes * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, teacher_mean: torch.Tensor) -> torch.Tensor:
        t_embed = t.float().unsqueeze(1) / max(1, self.timesteps - 1)
        feats = torch.cat([x_t, teacher_mean, t_embed], dim=1)
        return self.net(feats)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].unsqueeze(1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        teacher_mean: torch.Tensor,
    ) -> torch.Tensor:
        eps_pred = self.forward(x_t, torch.full((x_t.shape[0],), t, device=x_t.device), teacher_mean)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = (x_t - coef * eps_pred) / torch.sqrt(alpha)
        if t == 0:
            return mean
        noise = torch.randn_like(x_t)
        beta = self.betas[t]
        return mean + torch.sqrt(beta) * noise

    def denoise(self, x0: torch.Tensor, teacher_mean: torch.Tensor) -> torch.Tensor:
        x_t = x0
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t, t, teacher_mean)
        return torch.clamp(x_t, min=0.0)
