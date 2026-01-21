import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from anndata import read_h5ad
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from run_simple_imputers import gene_median_impute, svd_impute, knn_impute


def infer_dataset_id(path: str) -> str:
    """Infer dataset id from file name."""
    stem = Path(path).name
    dataset_id = stem.split(".")[0]
    if dataset_id.endswith("_raw") or dataset_id.endswith("_pre"):
        dataset_id = dataset_id[:-4]
    return dataset_id


def to_dense(matrix) -> np.ndarray:
    """Convert a dense or sparse matrix to a dense numpy array."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def load_counts(
    input_path: str,
    max_cells: Optional[int] = None,
    max_genes: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw counts from an AnnData file, optionally subsampling cells/genes.

    Returns:
        counts: float32 array (cells x genes)
        cell_ids: array of cell ids
        gene_ids: array of gene ids
        cell_idx: indices used for subsampling
        gene_idx: indices used for subsampling
    """
    adata = read_h5ad(input_path, backed=None)
    counts = to_dense(adata.X).astype(np.float32)
    cell_ids = np.array(adata.obs_names).astype(str)
    gene_ids = np.array(adata.var_names).astype(str)
    rng = np.random.default_rng(seed)
    cell_idx = np.arange(counts.shape[0])
    gene_idx = np.arange(counts.shape[1])
    if max_cells and counts.shape[0] > max_cells:
        cell_idx = rng.choice(counts.shape[0], size=max_cells, replace=False)
    if max_genes and counts.shape[1] > max_genes:
        gene_idx = rng.choice(counts.shape[1], size=max_genes, replace=False)
    counts = counts[cell_idx][:, gene_idx]
    cell_ids = cell_ids[cell_idx]
    gene_ids = gene_ids[gene_idx]
    return counts, cell_ids, gene_ids, cell_idx, gene_idx


def load_teacher_matrices(
    methods: List[str],
    output_root: str,
    dataset_id: str,
    disease: str,
    tissue: str,
    shape: Tuple[int, int],
    allow_compute: bool = False,
    counts: Optional[np.ndarray] = None,
    svd_components: int = 50,
    graph_neighbors: int = 30,
    graph_pca_components: int = 30,
    graph_include_self: bool = True,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load teacher imputation matrices from disk. Optionally compute simple baselines.

    The expected layout is:
        {output_root}/{method}/{dataset_id}/{disease}/{tissue}.npy
    """
    teachers: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for method in methods:
        path = Path(output_root) / method / dataset_id / disease / f"{tissue}.npy"
        if path.exists():
            matrix = np.load(path)
            if matrix.shape != shape:
                if allow_compute and counts is not None:
                    missing.append(method)
                    continue
                raise ValueError(f"Teacher {method} has shape {matrix.shape}, expected {shape}")
            teachers[method] = matrix.astype(np.float32)
        else:
            missing.append(method)

    if missing and allow_compute:
        if counts is None:
            raise ValueError("counts required to compute missing teachers")
        for method in missing:
            if method == "gene_median":
                teachers[method] = gene_median_impute(counts)
            elif method == "svd_impute":
                teachers[method] = svd_impute(counts, n_components=svd_components)
            elif method == "graph_smooth":
                teachers[method] = graph_smooth_impute(
                    counts,
                    n_neighbors=graph_neighbors,
                    n_components=graph_pca_components,
                    include_self=graph_include_self,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown teacher method '{method}' for on-the-fly compute")
    elif missing:
        raise FileNotFoundError(f"Missing teacher files for: {missing}")

    return teachers


def compute_teacher_from_counts(
    method: str,
    counts: np.ndarray,
    svd_components: int = 50,
    knn_neighbors: int = 5,
    graph_neighbors: int = 30,
    graph_pca_components: int = 30,
    graph_include_self: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Compute a lightweight teacher on a counts matrix."""
    if method == "gene_median":
        return gene_median_impute(counts)
    if method == "svd_impute":
        return svd_impute(counts, n_components=svd_components)
    if method == "knn_impute":
        return knn_impute(counts, n_neighbors=knn_neighbors)
    if method == "graph_smooth":
        return graph_smooth_impute(
            counts,
            n_neighbors=graph_neighbors,
            n_components=graph_pca_components,
            include_self=graph_include_self,
            seed=seed,
        )
    raise ValueError(f"Cannot recompute teacher for method '{method}'")


def log1p_cpm_dense(matrix: np.ndarray) -> np.ndarray:
    libsize = np.sum(matrix, axis=1)
    scale = 1e4 / (libsize + 1e-8)
    scaled = matrix * scale[:, None]
    return np.log1p(scaled)


def graph_smooth_impute(
    counts: np.ndarray,
    n_neighbors: int = 30,
    n_components: int = 30,
    include_self: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Graph smoothing teacher: build a kNN graph in PCA space and diffuse counts.
    Formula: T_graph = A * X, where A is row-normalized adjacency.
    """
    n_cells = counts.shape[0]
    if n_cells <= 1:
        return counts.astype(np.float32)
    k = max(1, min(n_neighbors, n_cells - 1))
    pca_components = min(n_components, counts.shape[1])

    norm = log1p_cpm_dense(counts.astype(np.float32))
    pca = PCA(n_components=pca_components, random_state=seed, svd_solver="randomized")
    embedding = pca.fit_transform(norm)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embedding)
    graph = nn.kneighbors_graph(embedding, n_neighbors=k, mode="connectivity")

    if include_self:
        import scipy.sparse as sp

        graph = graph + sp.eye(graph.shape[0], format="csr")

    graph = normalize(graph, norm="l1", axis=1)
    smoothed = graph.dot(counts.astype(np.float32))
    return np.asarray(smoothed, dtype=np.float32)


def subset_teachers(
    teachers: Dict[str, np.ndarray],
    cell_idx: np.ndarray,
    gene_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Apply cell/gene subset to each teacher matrix."""
    return {name: matrix[cell_idx][:, gene_idx] for name, matrix in teachers.items()}


def save_imputed_matrix(
    matrix: np.ndarray,
    output_root: str,
    method: str,
    dataset_id: str,
    disease: str,
    tissue: str,
) -> Path:
    """Persist imputed matrix to the standard output location."""
    dest = Path(output_root) / method / dataset_id / disease
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{tissue}.npy"
    np.save(path, matrix.astype(np.float32))
    return path


def save_metadata(
    output_dir: str,
    payload: Dict,
    name: str = "metadata.json",
) -> Path:
    """Save experiment metadata alongside outputs."""
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
