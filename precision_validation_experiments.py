import os
import json
import argparse
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from scipy.special import ndtr

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_message(message: str, log_file: Optional[Path] = None) -> None:
    """Log a short message to stdout and optionally append to a log file."""
    full_message = f"[eval] {message}"
    print(full_message)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(full_message + "\n")


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_marker_genes(markers_file: Optional[str]) -> Dict[str, List[str]]:
    """Load marker genes from a JSON file."""
    if not markers_file:
        return {}
    with open(markers_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    # Normalize casing for safer downstream matching
    normalized = {cell_type: [gene.upper() for gene in genes] for cell_type, genes in data.items()}
    return normalized


def log1p_cpm(matrix, inplace: bool = False):
    """
    Normalize a dense or sparse matrix to log1p(CPM).
    This mirrors the helper that exists in th_3_1.py but is kept local
    to avoid importing the entire training script.
    """
    is_sparse = hasattr(matrix, "tocsr")
    if is_sparse:
        libsize = np.asarray(matrix.sum(axis=1)).flatten()
    else:
        libsize = np.sum(matrix, axis=1)
    scale = 1e4 / (libsize + 1e-8)
    if is_sparse:
        import scipy.sparse as sp

        if not isinstance(matrix, sp.csr_matrix):
            matrix = matrix.tocsr()
        scaled = matrix.copy().astype(np.float64)
        for idx in range(matrix.shape[0]):
            start, end = matrix.indptr[idx], matrix.indptr[idx + 1]
            scaled.data[start:end] *= scale[idx]
        np.log1p(scaled.data, out=scaled.data)
        return scaled
    if inplace:
        for idx in range(matrix.shape[0]):
            matrix[idx, :] *= scale[idx]
        np.log1p(matrix, out=matrix)
        return matrix
    scaled = np.zeros_like(matrix)
    for idx in range(matrix.shape[0]):
        scaled[idx, :] = matrix[idx, :] * scale[idx]
    return np.log1p(scaled)


def standardize_names(names: List[str]) -> List[str]:
    """Upper-case and strip whitespace from gene or cell identifiers."""
    return [name.upper().strip() for name in names]


def infer_dataset_id(input_path: str) -> str:
    """Infer the dataset identifier from the h5ad file name."""
    stem = Path(input_path).name
    dataset_id = stem.split(".")[0]
    if dataset_id.endswith("_raw") or dataset_id.endswith("_pre"):
        dataset_id = dataset_id[:-4]
    return dataset_id


def get_imputation_file(
    root: str,
    dataset_id: str,
    method: str,
    disease: str,
    tissue: str,
) -> Optional[Path]:
    """
    Locate a cached imputation file. The layout mirrors the consensus scripts:
    output/{method}/{dataset}/{disease}/{tissue}.npy
    """
    candidate = Path(root) / method / dataset_id / disease
    if not candidate.exists():
        return None
    preferred = candidate / f"{tissue}.npy"
    if preferred.exists():
        return preferred
    variations = [
        f"{tissue} epithelium.npy",
        f"lamina propria of mucosa of {tissue}.npy",
        f"left {tissue}.npy",
        f"right {tissue}.npy",
        f"sigmoid {tissue}.npy",
    ]
    for variation in variations:
        alt = candidate / variation
        if alt.exists():
            return alt
    return None


def extract_value(matrix, row: int, col: int) -> float:
    """Fetch a single value from dense or sparse matrices."""
    value = matrix[row, col]
    if hasattr(value, "toarray"):
        return float(value.toarray()[0, 0])
    if np.isscalar(value):
        return float(value)
    return float(np.asarray(value).squeeze())


def extract_values(matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Vectorized helper that fetches many coordinates from a matrix."""
    values = np.zeros(len(rows), dtype=np.float32)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        values[idx] = extract_value(matrix, int(row), int(col))
    return values


def latent_probability_scores(
    mu_values: np.ndarray,
    var_values: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Probability score for expression > epsilon in log space:
        s = 1 - Phi((log1p(eps) - mu_log) / sqrt(var))
    """
    mu_log = np.log1p(np.clip(mu_values, a_min=0.0, a_max=None))
    var = np.clip(var_values, a_min=1e-6, a_max=None)
    z = (np.log1p(epsilon) - mu_log) / np.sqrt(var)
    scores = 1.0 - ndtr(z)
    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)


def latent_z_scores(
    mu_values: np.ndarray,
    var_values: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Z-score for expression > epsilon in log space:
        z = (mu_log - log1p(eps)) / sqrt(var)
    """
    mu_log = np.log1p(np.clip(mu_values, a_min=0.0, a_max=None))
    var = np.clip(var_values, a_min=1e-6, a_max=None)
    z = (mu_log - np.log1p(epsilon)) / np.sqrt(var)
    return np.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0)


def sample_zero_coordinates(
    matrix,
    max_samples: int,
    seed: int,
    zero_tolerance: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rejection-sample zero coordinates without materializing the full zero mask.
    Works for dense and CSR matrices.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_genes = matrix.shape
    rows, cols = [], []
    attempts, max_attempts = 0, max(10000, max_samples * 10)
    while len(rows) < max_samples and attempts < max_attempts:
        row = int(rng.integers(0, n_cells))
        col = int(rng.integers(0, n_genes))
        value = extract_value(matrix, row, col)
        if abs(value) <= zero_tolerance:
            rows.append(row)
            cols.append(col)
        attempts += 1
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)


def ensure_celltypes(
    adata,
    cluster_key: Optional[str] = None,
    max_clusters: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a cell-type/cluster label for every cell. If annotations exist,
    reuse them, otherwise derive data-driven clusters.
    """
    if cluster_key and cluster_key in adata.obs:
        return adata.obs[cluster_key].astype(str).values
    for key in ["Celltype", "cell_type", "celltype", "annotation"]:
        if key in adata.obs:
            return adata.obs[key].astype(str).values
    # Fall back to unsupervised clustering
    n_obs = adata.shape[0]
    n_clusters = min(max_clusters, max(2, n_obs // 200))
    sample_size = min(5000, n_obs)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_obs, sample_size, replace=False)
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    sample = matrix[sample_idx]
    reducer = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    reducer.fit(sample)
    assignments = reducer.predict(matrix)
    return np.array([f"Cluster_{label}" for label in assignments])


def dropout_fraction(matrix, rows: Optional[np.ndarray] = None, zero_tolerance: float = 1e-8) -> float:
    """Compute dropout fraction globally or for a slice without densifying large matrices."""
    if hasattr(matrix, "tocsr"):
        subset = matrix if rows is None else matrix[rows]
        total = subset.shape[0] * subset.shape[1]
        nonzero = subset.nnz
        return 1.0 - (nonzero / max(total, 1))
    subset = matrix if rows is None else matrix[rows]
    return float(np.mean(np.abs(subset) <= zero_tolerance))


def compute_dropout_by_celltype(matrix, celltypes: np.ndarray, zero_tolerance: float = 1e-8) -> Dict[str, float]:
    """Estimate dropout fraction per cell type without materializing dense matrices."""
    dropout = {}
    unique = np.unique(celltypes)
    for cell_type in unique:
        idx = np.where(celltypes == cell_type)[0]
        dropout[cell_type] = dropout_fraction(matrix, idx, zero_tolerance)
    return dropout


def load_validation_panels(
    panel_paths: List[str],
) -> pd.DataFrame:
    """
    Load and harmonize orthogonal validation panels (FISH, CITE-seq, ERCC spike-ins).
    Required columns (case insensitive):
        - gene / target
        - cell / cell_id / barcode
        - measurement / value / expression
    Optional columns:
        - assay (defaults to file stem)
        - label (binary precision target)
        - threshold (per-measurement cutoff)
    """
    frames = []
    for path in panel_paths:
        df = pd.read_csv(path)
        lower = {col.lower(): col for col in df.columns}
        gene_col = lower.get("gene") or lower.get("target")
        cell_col = lower.get("cell") or lower.get("cell_id") or lower.get("barcode")
        meas_col = (
            lower.get("measurement")
            or lower.get("value")
            or lower.get("expression")
            or lower.get("signal")
        )
        if not (gene_col and cell_col and meas_col):
            raise ValueError(f"Panel {path} missing required columns.")
        df = df.rename(
            columns={
                gene_col: "gene",
                cell_col: "cell_id",
                meas_col: "measurement",
            }
        )
        if "assay" not in df.columns:
            df["assay"] = Path(path).stem
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cell_id", "gene", "measurement", "assay"])
    combined = pd.concat(frames, ignore_index=True)
    combined["gene"] = combined["gene"].astype(str).str.upper()
    combined["cell_id"] = combined["cell_id"].astype(str)
    return combined


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


class PrecisionExperiment:
    """
    Precision-first consensus experiment that augments majority voting with:
        1) Reliability-weighted method fusion
        2) Rare-cell fairness constraints
        3) Orthogonal validation hooks (FISH/CITE/ERCC)
        4) Marker-placebo controls for false-positive estimation
        5) Threshold/grid sensitivity sweeps
        6) Divergence diagnostics across imputers
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "precision_experiment.log"
        self.dataset_id = infer_dataset_id(args.input_file)
        self.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        self.marker_genes = load_marker_genes(args.markers_file)
        self.validation_panels = (
            load_validation_panels(args.validation_panels) if args.validation_panels else pd.DataFrame()
        )
        self.assay_thresholds = json.loads(args.assay_thresholds) if args.assay_thresholds else {}
        self.adata = None
        self.celltypes = None
        self.original_matrix = None
        self.original_norm = None
        self.zero_rows = None
        self.zero_cols = None
        self.zero_values = None
        self.sample_celltypes = None
        self.sample_gene_names = None
        self.sample_gene_upper = None
        self.marker_mask = None
        self.method_values: Dict[str, np.ndarray] = {}
        self.method_scaled: Dict[str, np.ndarray] = {}
        self.method_variances: Dict[str, np.ndarray] = {}
        self.method_features: Dict[str, Dict[str, float]] = {}
        self.method_stats_frame = None
        self.safe_fusion_meta: Dict[str, Dict[str, float]] = {}
        self.sample_labels = None
        self.dropout_by_celltype = {}
        self.global_dropout = 0.0
        self.rare_cell_types: List[str] = []

    # --------------------
    # Loading and sampling
    # --------------------

    def load_anndata(self) -> None:
        log_message("Loading AnnData input...", self.log_file)
        backed = "r" if self.args.low_memory else None
        adata = read_h5ad(self.args.input_file, backed=backed)
        if self.args.max_cells and adata.n_obs > self.args.max_cells:
            idx = np.random.choice(adata.n_obs, self.args.max_cells, replace=False)
            adata = adata[idx].copy()
        if self.args.max_genes and adata.n_vars > self.args.max_genes:
            idx = np.random.choice(adata.n_vars, self.args.max_genes, replace=False)
            adata = adata[:, idx].copy()
        self.adata = adata
        self.original_matrix = adata.X
        matrix_copy = self.original_matrix.copy() if hasattr(self.original_matrix, "copy") else np.array(self.original_matrix)
        self.original_norm = log1p_cpm(matrix_copy)
        self.celltypes = ensure_celltypes(adata, cluster_key=self.args.cluster_key, seed=self.args.seed)
        self.sample_gene_names = np.array(adata.var_names)
        self.sample_gene_upper = standardize_names(self.sample_gene_names.tolist())
        self.cell_ids = np.array(adata.obs_names).astype(str)
        self.dropout_by_celltype = compute_dropout_by_celltype(self.original_matrix, self.celltypes)
        self.global_dropout = dropout_fraction(self.original_matrix, zero_tolerance=self.args.zero_tolerance)
        freq = pd.Series(self.celltypes).value_counts(normalize=True)
        self.rare_cell_types = freq[freq <= self.args.rare_cell_fraction].index.tolist()
        log_message(
            f"Loaded {adata.n_obs} cells, {adata.n_vars} genes. Rare cell types: {self.rare_cell_types}",
            self.log_file,
        )

    def sample_zero_entries(self) -> None:
        log_message("Sampling zero coordinates for evaluation...", self.log_file)
        rows, cols = sample_zero_coordinates(
            self.original_matrix,
            max_samples=self.args.max_zeros,
            seed=self.args.seed,
            zero_tolerance=self.args.zero_tolerance,
        )
        if len(rows) == 0:
            raise RuntimeError("Could not locate zero entries to analyze.")
        self.zero_rows, self.zero_cols = rows, cols
        self.zero_values = extract_values(self.original_matrix, rows, cols)
        self.sample_celltypes = self.celltypes[rows]
        self.sample_gene_names = self.adata.var_names[cols]
        self.sample_gene_upper = standardize_names(list(self.sample_gene_names))
        log_message(
            f"Collected {len(rows)} zero entries for downstream evaluation.",
            self.log_file,
        )

    # --------------------
    # Marker and labels
    # --------------------

    def build_marker_mask(self, marker_map: Optional[Dict[str, List[str]]] = None) -> np.ndarray:
        """Map each sampled zero to whether it belongs to a marker gene for that cell type."""
        if marker_map is None:
            marker_map = self.marker_genes
        marker_sets = {cell_type: set(genes) for cell_type, genes in marker_map.items()}
        mask = np.zeros(len(self.zero_rows), dtype=bool)
        if not marker_sets:
            return mask
        for idx, (cell_type, gene) in enumerate(zip(self.sample_celltypes, self.sample_gene_upper)):
            if gene in marker_sets.get(cell_type, set()):
                mask[idx] = True
        return mask

    def build_ground_truth_labels(self) -> None:
        """Attach orthogonal validation labels (if available) to sampled zeros."""
        if self.validation_panels.empty:
            self.sample_labels = np.full(len(self.zero_rows), np.nan)
            return
        panel = self.validation_panels.copy()
        threshold_default = self.args.validation_threshold
        if "label" in panel.columns:
            panel["label"] = panel["label"].astype(int)
        else:
            thresholds = {
                assay.upper(): float(self.assay_thresholds.get(assay.upper(), threshold_default))
                for assay in panel["assay"].astype(str).str.upper().unique()
            }
            panel["label"] = panel.apply(
                lambda row: int(row["measurement"] >= thresholds[row["assay"].upper()]), axis=1
            )
        lookup = {
            (row.cell_id, row.gene): (row.label, row.assay)
            for row in panel.itertuples(index=False)
        }
        labels = np.full(len(self.zero_rows), np.nan)
        assays = np.empty(len(self.zero_rows), dtype=object)
        for idx, (cell_idx, gene) in enumerate(
            zip(self.zero_rows, self.sample_gene_upper)
        ):
            key = (self.cell_ids[cell_idx], gene)
            if key in lookup:
                labels[idx] = lookup[key][0]
                assays[idx] = lookup[key][1]
        self.sample_labels = labels
        self.sample_assays = assays
        labeled = np.sum(~np.isnan(labels))
        log_message(
            f"Labeled {labeled} sampled zeros with orthogonal evidence.",
            self.log_file,
        )

    # --------------------
    # Imputation loading
    # --------------------

    def load_method_values(self) -> None:
        rows, cols = self.zero_rows, self.zero_cols
        for method in self.methods:
            path = get_imputation_file(
                root=self.args.imputation_root,
                dataset_id=self.dataset_id,
                method=method,
                disease=self.args.disease,
                tissue=self.args.tissue,
            )
            if path is None:
                log_message(f"Skipping {method}: no cached imputation found.", self.log_file)
                continue
            log_message(f"Loading imputed matrix for {method} from {path}", self.log_file)
            matrix = np.load(path, mmap_mode="r" if self.args.low_memory else None)
            if matrix.shape != self.original_matrix.shape:
                raise ValueError(f"Shape mismatch for {method}: expected {self.original_matrix.shape}, got {matrix.shape}")
            matrix = np.array(matrix, dtype=np.float32)
            np.clip(matrix, a_min=0.0, a_max=None, out=matrix)
            var_path = path.with_name(f"{path.stem}_var.npy")
            use_variance = self.args.latent_use_prob and var_path.exists()
            if self.args.latent_use_prob and not var_path.exists():
                log_message(f"Variance scoring requested but var file missing at {var_path}", self.log_file)
            if use_variance:
                var_matrix = np.load(var_path, mmap_mode="r" if self.args.low_memory else None)
                if var_matrix.shape != matrix.shape:
                    raise ValueError(
                        f"Var shape mismatch for {method}: expected {matrix.shape}, got {var_matrix.shape}"
                    )
                mu_vals = extract_values(matrix, rows, cols)
                var_vals = extract_values(var_matrix, rows, cols)
                self.method_variances[method] = var_vals.astype(np.float32)
                if self.args.latent_score_mode == "prob":
                    values = latent_probability_scores(mu_vals, var_vals, epsilon=self.args.latent_prob_eps)
                    log_message(
                        f"Using latent probability scores for {method} with eps={self.args.latent_prob_eps}",
                        self.log_file,
                    )
                else:
                    values = latent_z_scores(mu_vals, var_vals, epsilon=self.args.latent_prob_eps)
                    log_message(
                        f"Using latent z-scores for {method} with eps={self.args.latent_prob_eps}",
                        self.log_file,
                    )
            else:
                norm = log1p_cpm(matrix)
                values = extract_values(norm, rows, cols)
            self.method_values[method] = values.astype(np.float32)
        if not self.method_values:
            raise RuntimeError("No imputation methods were successfully loaded.")
        log_message(f"Collected values for methods: {list(self.method_values.keys())}", self.log_file)

    # --------------------
    # Reliability features
    # --------------------

    def compute_method_features(self) -> None:
        self.marker_mask = self.build_marker_mask()
        control_mask = ~self.marker_mask if np.any(self.marker_mask) else np.ones_like(self.marker_mask, dtype=bool)
        label_mask = ~np.isnan(self.sample_labels) if self.sample_labels is not None else np.array([False] * len(self.zero_rows))

        stacked = []
        feature_rows = []

        for method, values in self.method_values.items():
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
            self.method_scaled[method] = scaled
            stacked.append(scaled)

            marker_gain = (
                float(np.mean(values[self.marker_mask] >= self.args.marker_delta))
                if np.any(self.marker_mask)
                else 0.0
            )
            control_fp = (
                float(np.mean(values[control_mask] >= self.args.marker_delta))
                if np.any(control_mask)
                else 1.0
            )
            rare_mask = np.isin(self.sample_celltypes, self.rare_cell_types)
            rare_support = (
                float(np.mean(values[rare_mask] >= self.args.rare_cell_floor))
                if np.any(rare_mask)
                else 0.0
            )
            if np.any(label_mask):
                y_true = self.sample_labels[label_mask]
                y_scores = values[label_mask]
                try:
                    ap = average_precision_score(y_true, y_scores)
                except ValueError:
                    ap = 0.0
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    auc = 0.5
            else:
                ap, auc = 0.0, 0.5

            feature_rows.append(
                {
                    "method": method,
                    "marker_gain": marker_gain,
                    "control_fp": control_fp,
                    "rare_support": rare_support,
                    "ground_truth_ap": ap,
                    "ground_truth_auc": auc,
                }
            )

        stacked_matrix = np.vstack(stacked)
        mean_vector = np.mean(stacked_matrix, axis=0)
        for row in feature_rows:
            method = row["method"]
            disagreement = float(np.mean(np.abs(self.method_scaled[method] - mean_vector)))
            row["disagreement_penalty"] = disagreement
            self.method_features[method] = row

        self.method_stats_frame = pd.DataFrame(feature_rows)
        stats_path = self.output_dir / "method_reliability.csv"
        self.method_stats_frame.to_csv(stats_path, index=False)
        log_message(f"Saved method reliability features to {stats_path}", self.log_file)

    def add_safe_fusion_method(self) -> None:
        method = self.args.safe_fusion_method.strip()
        if not method or not self.args.safe_fusion_teachers:
            return
        if method not in self.method_values:
            log_message(f"Safe-fusion skipped: {method} not found in loaded methods.", self.log_file)
            return
        teachers = [m.strip() for m in self.args.safe_fusion_teachers.split(",") if m.strip()]
        teachers = [m for m in teachers if m and m != method]
        if not teachers:
            log_message("Safe-fusion skipped: no valid teacher list after filtering.", self.log_file)
            return
        missing = [m for m in teachers if m not in self.method_values]
        if missing:
            log_message(f"Safe-fusion skipped: missing teacher scores for {missing}.", self.log_file)
            return
        if self.args.safe_fusion_conf_mode == "var":
            if method not in self.method_variances:
                log_message(f"Safe-fusion skipped: variance scores missing for {method}.", self.log_file)
                return
            var_vals = self.method_variances[method]
            conf = -np.log(np.clip(var_vals, a_min=1e-8, a_max=None))
            conf = np.nan_to_num(conf, nan=-1e6, neginf=-1e6, posinf=1e6)
        else:
            raise ValueError(f"Unsupported safe-fusion confidence mode: {self.args.safe_fusion_conf_mode}")

        if self.args.safe_fusion_conf_threshold is not None:
            threshold = float(self.args.safe_fusion_conf_threshold)
        else:
            quantile = float(self.args.safe_fusion_conf_quantile)
            if not 0.0 < quantile < 1.0:
                raise ValueError("--safe_fusion_conf_quantile must be between 0 and 1.")
            threshold = float(np.quantile(conf, quantile))

        score_fuse = self.method_values[method]
        teacher_matrix = np.vstack([self.method_values[m] for m in teachers])
        score_best = np.max(teacher_matrix, axis=0)
        safe_scores = np.where(conf >= threshold, score_fuse, score_best)
        name = self.args.safe_fusion_name.strip() or f"safe_{method}"
        self.method_values[name] = safe_scores.astype(np.float32)
        coverage_fusion = float(np.mean(conf >= threshold))
        self.safe_fusion_meta[name] = {
            "coverage_fusion": coverage_fusion,
            "coverage_fallback": float(1.0 - coverage_fusion),
            "confidence_threshold": float(threshold),
        }
        log_message(
            f"Added safe-fusion method {name} using {method} (conf >= {threshold:.4f}, "
            f"teachers={teachers}).",
            self.log_file,
        )

    def compute_weights(self, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Convert method features into reliability weights via a softmax."""
        params = {
            "alpha_marker": self.args.alpha_marker,
            "alpha_ground": self.args.alpha_ground,
            "alpha_rare": self.args.alpha_rare,
            "alpha_penalty": self.args.alpha_penalty,
            "alpha_fp": self.args.alpha_fp,
        }
        if overrides:
            params.update({k: v for k, v in overrides.items() if k in params})
        scores = []
        for method, feats in self.method_features.items():
            score = (
                params["alpha_marker"] * feats["marker_gain"]
                + params["alpha_ground"] * feats["ground_truth_ap"]
                + params["alpha_rare"] * feats["rare_support"]
                - params["alpha_penalty"] * feats["disagreement_penalty"]
                - params["alpha_fp"] * feats["control_fp"]
            )
            scores.append((method, score))
        weights = np.array([np.exp(score) for _, score in scores])
        weights /= np.sum(weights)
        return {method: float(weight) for (method, _), weight in zip(scores, weights)}

    # --------------------
    # Aggregation + metrics
    # --------------------

    def build_thresholds(self, config: Dict[str, float]) -> np.ndarray:
        base = config.get("decision_threshold", self.args.decision_threshold)
        dropout_weight = config.get("dropout_weight", self.args.dropout_weight)
        rare_boost = config.get("rare_cell_boost", self.args.rare_cell_boost)
        per_sample = np.full(len(self.sample_celltypes), base, dtype=np.float32)
        for cell_type in np.unique(self.sample_celltypes):
            idx = np.where(self.sample_celltypes == cell_type)[0]
            if idx.size == 0:
                continue
            cell_dropout = self.dropout_by_celltype.get(cell_type, self.global_dropout)
            shift = dropout_weight * (cell_dropout - self.global_dropout)
            if cell_type in self.rare_cell_types:
                shift -= rare_boost
            per_sample[idx] = base + shift
        return per_sample

    def aggregate_probabilities(
        self,
        weights: Dict[str, float],
        config: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Combine scaled method outputs into consensus probabilities."""
        config = config or {}
        score = np.zeros(len(self.zero_rows), dtype=np.float32)
        for method, weight in weights.items():
            score += weight * np.nan_to_num(self.method_scaled[method], nan=0.0)
        thresholds = self.build_thresholds(config)
        margin = score - thresholds
        probabilities = 1.0 / (1.0 + np.exp(-margin))
        return np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)

    def evaluate_predictions(
        self,
        probabilities: np.ndarray,
        config: Dict[str, float],
        label_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute downstream metrics using available orthogonal labels."""
        decision = config.get("fill_cutoff", self.args.fill_cutoff)
        labels_available = label_mask if label_mask is not None else ~np.isnan(self.sample_labels)
        metrics = {}
        if np.any(labels_available):
            y_true = self.sample_labels[labels_available]
            y_scores = probabilities[labels_available]
            metrics.update(self._compute_label_metrics(y_true, y_scores, decision, include_curves=True))
        metrics["fill_fraction"] = float(np.mean(probabilities >= decision))
        rare_mask = np.isin(self.sample_celltypes, self.rare_cell_types)
        if np.any(rare_mask):
            metrics["rare_cell_fill_fraction"] = float(np.mean(probabilities[rare_mask] >= decision))
        return metrics

    def _compute_label_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        decision: float,
        include_curves: bool = False,
    ) -> Dict[str, float]:
        """Scalar metrics (optionally curves) for a label/score pair."""
        metrics: Dict[str, float] = {}
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = average_precision_score(labels, scores)
        try:
            roc = roc_auc_score(labels, scores)
        except ValueError:
            roc = 0.5
        if include_curves:
            metrics["precision_curve"] = precision.tolist()
            metrics["recall_curve"] = recall.tolist()
        preds = (scores >= decision).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision_point = tp / (tp + fp + 1e-8)
        recall_point = tp / (tp + fn + 1e-8)
        f1 = 2 * precision_point * recall_point / (precision_point + recall_point + 1e-8)
        metrics.update(
            {
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc),
                "precision_at_threshold": float(precision_point),
                "recall_at_threshold": float(recall_point),
                "f1_at_threshold": float(f1),
            }
        )
        return metrics

    @staticmethod
    def _compute_binary_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """Precision/recall/f1 from binary predictions."""
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def run_operating_point_sweep(self) -> pd.DataFrame:
        """Compute fill-rate matched operating points for each method."""
        if not self.method_values:
            raise RuntimeError("No method values loaded for operating point sweep.")
        if self.marker_mask is None:
            self.marker_mask = self.build_marker_mask()
        control_mask = ~self.marker_mask if np.any(self.marker_mask) else np.ones_like(self.marker_mask, dtype=bool)
        rare_mask = np.isin(self.sample_celltypes, self.rare_cell_types)
        label_mask = ~np.isnan(self.sample_labels) if self.sample_labels is not None else np.array([False] * len(self.zero_rows))
        records = []
        for method, values in self.method_values.items():
            for target in self.args.fill_rate_targets:
                threshold = float(np.quantile(values, 1.0 - target))
                preds = values >= threshold
                fill_rate = float(np.mean(preds))
                metrics = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
                if np.any(label_mask):
                    metrics = self._compute_binary_metrics(self.sample_labels[label_mask], preds[label_mask].astype(int))
                control_fp = float(np.mean(preds[control_mask])) if np.any(control_mask) else np.nan
                rare_support = float(np.mean(preds[rare_mask])) if np.any(rare_mask) else np.nan
                records.append(
                    {
                        "method": method,
                        "fill_rate_target": float(target),
                        "threshold": threshold,
                        "fill_rate_actual": fill_rate,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "control_fp": control_fp,
                        "rare_support": rare_support,
                        "coverage_fusion": self.safe_fusion_meta.get(method, {}).get("coverage_fusion", np.nan),
                        "coverage_fallback": self.safe_fusion_meta.get(method, {}).get("coverage_fallback", np.nan),
                    }
                )
        df = pd.DataFrame(records)
        path = self.output_dir / "method_operating_points.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved operating point sweep to {path}", self.log_file)
        return df

    def bootstrap_method_metrics(self) -> Optional[pd.DataFrame]:
        """Bootstrap AP/AUC and operating point metrics for each method."""
        label_mask = ~np.isnan(self.sample_labels)
        if not np.any(label_mask):
            log_message("Skipping per-method bootstrap: no orthogonal labels available.", self.log_file)
            return None
        rng = np.random.default_rng(self.args.seed + 47)
        labeled_idx = np.where(label_mask)[0]
        if labeled_idx.size == 0:
            return None
        target = self.args.fill_rate_bootstrap_target
        rounds = self.args.method_bootstrap_rounds
        rows = []
        for method, values in self.method_values.items():
            scores = values
            threshold = float(np.quantile(scores, 1.0 - target))
            y_true_full = self.sample_labels[label_mask]
            y_scores_full = scores[label_mask]
            preds_full = (y_scores_full >= threshold).astype(int)
            observed = {}
            try:
                observed["ap"] = float(average_precision_score(y_true_full, y_scores_full))
            except ValueError:
                observed["ap"] = np.nan
            try:
                observed["auc"] = float(roc_auc_score(y_true_full, y_scores_full))
            except ValueError:
                observed["auc"] = np.nan
            obs_bin = self._compute_binary_metrics(y_true_full, preds_full)
            observed["precision"] = obs_bin["precision"]
            observed["f1"] = obs_bin["f1"]

            samples = {"ap": [], "auc": [], "precision": [], "f1": []}
            for _ in range(rounds):
                sample_idx = rng.choice(labeled_idx, size=len(labeled_idx), replace=True)
                y_true = self.sample_labels[sample_idx]
                y_scores = scores[sample_idx]
                preds = (y_scores >= threshold).astype(int)
                try:
                    ap = average_precision_score(y_true, y_scores)
                except ValueError:
                    ap = np.nan
                try:
                    auc = roc_auc_score(y_true, y_scores)
                except ValueError:
                    auc = np.nan
                bin_metrics = self._compute_binary_metrics(y_true, preds)
                samples["ap"].append(ap)
                samples["auc"].append(auc)
                samples["precision"].append(bin_metrics["precision"])
                samples["f1"].append(bin_metrics["f1"])

            for metric, values_list in samples.items():
                vals = np.array(values_list, dtype=np.float64)
                rows.append(
                    {
                        "method": method,
                        "fill_rate_target": float(target),
                        "metric": metric,
                        "observed": float(observed.get(metric, np.nan)),
                        "mean": float(np.nanmean(vals)),
                        "p2_5": float(np.nanpercentile(vals, 2.5)),
                        "p97_5": float(np.nanpercentile(vals, 97.5)),
                        "rounds": int(rounds),
                    }
                )
        df = pd.DataFrame(rows)
        path = self.output_dir / "method_bootstrap_metrics.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved per-method bootstrap metrics to {path}", self.log_file)
        return df

    # --------------------
    # Diagnostics
    # --------------------

    def run_divergence_report(self) -> None:
        rows = []
        methods = list(self.method_values.keys())
        for a, b in combinations(methods, 2):
            values_a, values_b = self.method_values[a], self.method_values[b]
            corr = np.corrcoef(values_a, values_b)[0, 1]
            mad = float(np.mean(np.abs(values_a - values_b)))
            rows.append({"method_a": a, "method_b": b, "pearson": corr, "mad": mad})
        df = pd.DataFrame(rows)
        path = self.output_dir / "method_divergence.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved method divergence report to {path}", self.log_file)

    def run_placebo_analysis(self, weights: Dict[str, float]) -> Optional[Dict[str, float]]:
        if not self.marker_genes:
            log_message("Skipping placebo analysis: no marker definitions provided.", self.log_file)
            return None
        rng = np.random.default_rng(self.args.seed + 13)
        cell_types = list(self.marker_genes.keys())
        shuffled = cell_types.copy()
        rng.shuffle(shuffled)
        placebo_map = {cell_type: self.marker_genes[shuffled_idx] for cell_type, shuffled_idx in zip(cell_types, shuffled)}
        placebo_mask = self.build_marker_mask(placebo_map)
        control_mask = ~placebo_mask if np.any(placebo_mask) else np.ones_like(placebo_mask, dtype=bool)
        placebo_features = {}
        for method, values in self.method_values.items():
            if method not in self.method_features:
                continue
            marker_gain = (
                float(np.mean(values[placebo_mask] >= self.args.marker_delta))
                if np.any(placebo_mask)
                else 0.0
            )
            control_fp = (
                float(np.mean(values[control_mask] >= self.args.marker_delta))
                if np.any(control_mask)
                else 1.0
            )
            placebo_features[method] = {**self.method_features[method]}
            placebo_features[method]["marker_gain"] = marker_gain
            placebo_features[method]["control_fp"] = control_fp
        original_features = self.method_features
        self.method_features = placebo_features
        placebo_weights = self.compute_weights()
        placebo_probs = self.aggregate_probabilities(placebo_weights)
        placebo_metrics = self.evaluate_predictions(placebo_probs, config={})
        # Restore original features for downstream work
        self.method_features = original_features
        return {
            "placebo_weights": placebo_weights,
            "placebo_metrics": placebo_metrics,
            "original_weights": weights,
        }

    def run_label_shuffle(self, probabilities: np.ndarray) -> Optional[Dict[str, float]]:
        """Shuffle orthogonal labels to estimate false-positive performance under adversarial priors."""
        label_mask = ~np.isnan(self.sample_labels)
        if not np.any(label_mask):
            log_message("Skipping label-shuffle analysis: no orthogonal labels available.", self.log_file)
            return None
        rng = np.random.default_rng(self.args.seed + 17)
        y_true = self.sample_labels[label_mask]
        y_scores = probabilities[label_mask]
        baseline_metrics = self._compute_label_metrics(y_true, y_scores, self.args.fill_cutoff, include_curves=False)
        keys = ["pr_auc", "precision_at_threshold", "recall_at_threshold", "f1_at_threshold", "roc_auc"]
        distributions = {key: [] for key in keys}
        for _ in range(self.args.shuffle_rounds):
            shuffled = rng.permutation(y_true)
            shuffled_metrics = self._compute_label_metrics(shuffled, y_scores, self.args.fill_cutoff, include_curves=False)
            for key in keys:
                distributions[key].append(shuffled_metrics[key])
        summary = {
            "rounds": self.args.shuffle_rounds,
            "observed": baseline_metrics,
            "shuffle_baseline": {
                key: {
                    "mean": float(np.nanmean(distributions[key])),
                    "p2_5": float(np.nanpercentile(distributions[key], 2.5)),
                    "p97_5": float(np.nanpercentile(distributions[key], 97.5)),
                }
                for key in keys
            },
        }
        path = self.output_dir / "label_shuffle_metrics.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({**summary, "distributions": distributions}, handle, indent=2)
        log_message(f"Saved label-shuffle placebo metrics to {path}", self.log_file)
        return summary

    def compute_per_celltype_metrics(self, probabilities: np.ndarray) -> pd.DataFrame:
        """Per-cell-type precision/recall and fill fractions using orthogonal labels."""
        label_mask = ~np.isnan(self.sample_labels)
        celltypes = np.unique(self.sample_celltypes[label_mask]) if np.any(label_mask) else []
        rows = []
        for cell_type in celltypes:
            subset_mask = self.sample_celltypes == cell_type
            labeled_mask = label_mask & subset_mask
            labeled_count = int(np.sum(labeled_mask))
            metrics = {}
            if labeled_count >= self.args.min_labels_per_celltype:
                metrics = self._compute_label_metrics(
                    self.sample_labels[labeled_mask],
                    probabilities[labeled_mask],
                    self.args.fill_cutoff,
                    include_curves=False,
                )
            fill_fraction = float(np.mean(probabilities[subset_mask] >= self.args.fill_cutoff))
            row = {
                "cell_type": cell_type,
                "is_rare": cell_type in self.rare_cell_types,
                "labeled_count": labeled_count,
                "fill_fraction": fill_fraction,
                "pr_auc": metrics.get("pr_auc", np.nan),
                "roc_auc": metrics.get("roc_auc", np.nan),
                "precision_at_threshold": metrics.get("precision_at_threshold", np.nan),
                "recall_at_threshold": metrics.get("recall_at_threshold", np.nan),
                "f1_at_threshold": metrics.get("f1_at_threshold", np.nan),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        path = self.output_dir / "per_celltype_metrics.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved per-cell-type metrics to {path}", self.log_file)
        return df

    def bootstrap_metrics(self, probabilities: np.ndarray) -> Optional[Dict[str, Dict[str, float]]]:
        """Bootstrap confidence intervals for key metrics."""
        label_mask = ~np.isnan(self.sample_labels)
        if not np.any(label_mask):
            log_message("Skipping bootstrap metrics: no orthogonal labels available.", self.log_file)
            return None
        rng = np.random.default_rng(self.args.seed + 31)
        labeled_idx = np.where(label_mask)[0]
        metric_keys = ["pr_auc", "precision_at_threshold", "recall_at_threshold", "f1_at_threshold", "roc_auc"]
        metric_samples = {key: [] for key in metric_keys}
        fill_mask = probabilities >= self.args.fill_cutoff
        fill_indices = np.arange(len(fill_mask))
        fill_samples: List[float] = []
        for _ in range(self.args.bootstrap_rounds):
            sample_idx = rng.choice(labeled_idx, size=len(labeled_idx), replace=True)
            y_true = self.sample_labels[sample_idx]
            y_scores = probabilities[sample_idx]
            metrics = self._compute_label_metrics(y_true, y_scores, self.args.fill_cutoff, include_curves=False)
            for key in metric_keys:
                metric_samples[key].append(metrics[key])
            sampled_fills = np.mean(fill_mask[rng.choice(fill_indices, size=len(fill_indices), replace=True)])
            fill_samples.append(float(sampled_fills))
        summary = {
            key: {
                "mean": float(np.nanmean(vals)),
                "p2_5": float(np.nanpercentile(vals, 2.5)),
                "p97_5": float(np.nanpercentile(vals, 97.5)),
            }
            for key, vals in metric_samples.items()
        }
        summary["fill_fraction"] = {
            "mean": float(np.nanmean(fill_samples)),
            "p2_5": float(np.nanpercentile(fill_samples, 2.5)),
            "p97_5": float(np.nanpercentile(fill_samples, 97.5)),
        }
        path = self.output_dir / "bootstrap_metrics.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        log_message(f"Saved bootstrap confidence intervals to {path}", self.log_file)
        return summary

    def save_weight_summary(self, weights: Dict[str, float]) -> Path:
        """Persist weight ranking alongside reliability features."""
        rows = []
        for method, feats in self.method_features.items():
            row = {**feats, "weight": weights.get(method, 0.0)}
            rows.append(row)
        df = pd.DataFrame(rows)
        path = self.output_dir / "method_weight_summary.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved method weight summary to {path}", self.log_file)
        return path

    def run_sensitivity_analysis(self) -> pd.DataFrame:
        grid = self.args.sensitivity_grid
        if not grid:
            # Default sweep
            grid = [
                {"decision_threshold": thr, "rare_cell_boost": boost}
                for thr in [0.45, 0.55, 0.65]
                for boost in [0.05, 0.1, 0.2]
            ]
        records = []
        for idx, config in enumerate(grid):
            alpha_overrides = {k: v for k, v in config.items() if k.startswith("alpha_")}
            weights = self.compute_weights(alpha_overrides if alpha_overrides else None)
            probs = self.aggregate_probabilities(weights, config)
            metrics = self.evaluate_predictions(probs, config)
            records.append({**config, **metrics})
            log_message(f"Sensitivity combo {idx+1}/{len(grid)} evaluated.", self.log_file)
        df = pd.DataFrame(records)
        path = self.output_dir / "sensitivity_results.csv"
        df.to_csv(path, index=False)
        log_message(f"Saved sensitivity sweep to {path}", self.log_file)
        return df

    # --------------------
    # Main orchestration
    # --------------------

    def run(self) -> None:
        self.load_anndata()
        self.sample_zero_entries()
        self.build_ground_truth_labels()
        self.load_method_values()
        self.compute_method_features()
        self.add_safe_fusion_method()
        operating_points = self.run_operating_point_sweep()
        per_method_bootstrap = self.bootstrap_method_metrics()
        default_weights = self.compute_weights()
        weight_summary_path = self.save_weight_summary(default_weights)
        probabilities = self.aggregate_probabilities(default_weights, {})
        summary_metrics = self.evaluate_predictions(probabilities, {})
        label_shuffle = self.run_label_shuffle(probabilities)
        per_cell = self.compute_per_celltype_metrics(probabilities)
        bootstrap = self.bootstrap_metrics(probabilities)
        placebo = self.run_placebo_analysis(default_weights)
        sensitivity = self.run_sensitivity_analysis()
        self.run_divergence_report()

        # Persist outputs
        np.save(self.output_dir / "consensus_probabilities.npy", probabilities)
        with open(self.output_dir / "weights.json", "w", encoding="utf-8") as handle:
            json.dump(default_weights, handle, indent=2)
        summary_bundle = {
            "dataset": self.dataset_id,
            "methods": list(self.method_values.keys()),
            "summary_metrics": summary_metrics,
            "rare_cell_types": self.rare_cell_types,
            "weight_summary_file": str(weight_summary_path),
            "per_celltype_metrics_file": str(self.output_dir / "per_celltype_metrics.csv"),
            "operating_points_file": str(self.output_dir / "method_operating_points.csv"),
            "bootstrap_methods_file": str(self.output_dir / "method_bootstrap_metrics.csv")
            if per_method_bootstrap is not None
            else None,
        }
        if placebo:
            summary_bundle["placebo"] = placebo
        if label_shuffle:
            summary_bundle["label_shuffle"] = label_shuffle
        if bootstrap:
            summary_bundle["bootstrap"] = bootstrap
        summary_bundle["sensitivity_rows"] = len(sensitivity)
        with open(self.output_dir / "precision_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary_bundle, handle, indent=2)
        if np.any(~np.isnan(self.sample_labels)):
            mask = ~np.isnan(self.sample_labels)
            y_true = self.sample_labels[mask]
            y_scores = probabilities[mask]
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            pr_df = pd.DataFrame({"precision": precision, "recall": recall})
            pr_df.to_csv(self.output_dir / "precision_recall_curve.csv", index=False)
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
            roc_df.to_csv(self.output_dir / "roc_curve.csv", index=False)
        log_message("Precision-first experiment complete.", self.log_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precision-first evaluation suite for consensus imputation improvements.",
    )
    parser.add_argument("--input_file", required=True, help="Path to the AnnData (.h5ad) file.")
    parser.add_argument("--disease", required=True, help="Disease identifier used in cached outputs.")
    parser.add_argument("--tissue", required=True, help="Tissue identifier used in cached outputs.")
    parser.add_argument("--methods", type=str, default="SAUCIE,MAGIC,deepImpute,scScope,scVI")
    parser.add_argument("--markers_file", type=str, default=None, help="JSON file containing marker genes.")
    parser.add_argument("--validation_panels", nargs="*", default=[], help="List of CSV files with orthogonal measurements (FISH, CITE, ERCC, etc.).")
    parser.add_argument("--assay_thresholds", type=str, default="", help="JSON string mapping assay names to positive thresholds.")
    parser.add_argument("--validation_threshold", type=float, default=0.5, help="Default measurement threshold when per-assay thresholds are unavailable.")
    parser.add_argument("--marker_delta", type=float, default=0.25, help="Minimum imputed expression to count as a recovered marker.")
    parser.add_argument("--rare_cell_floor", type=float, default=0.2, help="Minimum value to consider a rare-cell recovery.")
    parser.add_argument("--rare_cell_fraction", type=float, default=0.05, help="Fractional cutoff to treat clusters as rare.")
    parser.add_argument("--imputation_root", type=str, default="output", help="Root directory that stores per-method imputed arrays.")
    parser.add_argument("--output_dir", type=str, default="./precision_suite", help="Where to store consolidated artefacts.")
    parser.add_argument("--cluster_key", type=str, default=None, help="Optional adata.obs column with cell-type annotations.")
    parser.add_argument("--max_cells", type=int, default=15000, help="Optional max number of cells to keep for analysis.")
    parser.add_argument("--max_genes", type=int, default=None, help="Optional max number of genes to keep for analysis.")
    parser.add_argument("--max_zeros", type=int, default=200000, help="Number of zero entries to evaluate.")
    parser.add_argument("--zero_tolerance", type=float, default=1e-8, help="Tolerance for declaring an entry as zero.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--low_memory", action="store_true", help="Use mmap where possible to lower peak memory.")
    parser.add_argument("--shuffle_rounds", type=int, default=256, help="Number of shuffles for adversarial label placebo.")
    parser.add_argument("--bootstrap_rounds", type=int, default=200, help="Bootstrap rounds for confidence intervals.")
    parser.add_argument("--min_labels_per_celltype", type=int, default=5, help="Require this many labeled zeros before reporting per-cell-type metrics.")
    parser.add_argument("--alpha_marker", type=float, default=1.2)
    parser.add_argument("--alpha_ground", type=float, default=1.0)
    parser.add_argument("--alpha_rare", type=float, default=0.6)
    parser.add_argument("--alpha_penalty", type=float, default=0.8)
    parser.add_argument("--alpha_fp", type=float, default=0.5)
    parser.add_argument("--decision_threshold", type=float, default=0.6, help="Base threshold before fairness adjustments.")
    parser.add_argument("--fill_cutoff", type=float, default=0.6, help="Final cutoff used to declare an imputed fill.")
    parser.add_argument("--dropout_weight", type=float, default=0.3, help="How aggressively dropout differentials shift thresholds.")
    parser.add_argument("--rare_cell_boost", type=float, default=0.15, help="Bias reduction for rare cell types.")
    parser.add_argument(
        "--latent_use_prob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use latent_truth variance to compute probability scores.",
    )
    parser.add_argument(
        "--latent_prob_eps",
        type=float,
        default=1e-3,
        help="Epsilon threshold for P(expr > eps) probability scores.",
    )
    parser.add_argument(
        "--latent_score_mode",
        choices=["prob", "z"],
        default="z",
        help="Score mode for latent_truth when using variance (probability or z-score).",
    )
    parser.add_argument(
        "--fill_rate_targets",
        type=str,
        default="0.02,0.04,0.06,0.08,0.081,0.10,0.12",
        help="Comma-separated fill-rate targets for operating points.",
    )
    parser.add_argument(
        "--fill_rate_bootstrap_target",
        type=float,
        default=0.081,
        help="Fill-rate target used for per-method bootstrap metrics.",
    )
    parser.add_argument(
        "--method_bootstrap_rounds",
        type=int,
        default=500,
        help="Bootstrap rounds for per-method AP/AUC and operating point metrics.",
    )
    parser.add_argument(
        "--safe_fusion_method",
        type=str,
        default="",
        help="Method name to wrap with safe-fusion logic (e.g., latent_truth_scvi_select).",
    )
    parser.add_argument(
        "--safe_fusion_teachers",
        type=str,
        default="",
        help="Comma-separated teacher methods used as fallback (e.g., scVI,graph_smooth).",
    )
    parser.add_argument(
        "--safe_fusion_conf_mode",
        choices=["var"],
        default="var",
        help="Confidence metric for safe-fusion (currently only variance-based).",
    )
    parser.add_argument(
        "--safe_fusion_conf_quantile",
        type=float,
        default=0.7,
        help="Quantile threshold for confidence when no absolute threshold is provided.",
    )
    parser.add_argument(
        "--safe_fusion_conf_threshold",
        type=float,
        default=None,
        help="Absolute confidence threshold; overrides quantile when set.",
    )
    parser.add_argument(
        "--safe_fusion_name",
        type=str,
        default="",
        help="Name used to label the safe-fusion method in outputs.",
    )
    parser.add_argument("--sensitivity_grid", type=str, default="", help="Optional JSON file containing a list of sensitivity configurations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sensitivity_grid:
        if os.path.isfile(args.sensitivity_grid):
            with open(args.sensitivity_grid, "r", encoding="utf-8") as handle:
                args.sensitivity_grid = json.load(handle)
        else:
            args.sensitivity_grid = json.loads(args.sensitivity_grid)
    if isinstance(args.fill_rate_targets, str):
        args.fill_rate_targets = [float(x.strip()) for x in args.fill_rate_targets.split(",") if x.strip()]
    experiment = PrecisionExperiment(args)
    experiment.run()


if __name__ == "__main__":
    main()
