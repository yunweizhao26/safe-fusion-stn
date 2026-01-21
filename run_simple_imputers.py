import argparse
from pathlib import Path
from typing import List

import numpy as np
from anndata import read_h5ad
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


def log(msg: str) -> None:
    print(f"[impute] {msg}")


def to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def gene_median_impute(X: np.ndarray, zero_tol: float = 1e-8) -> np.ndarray:
    log("Running gene median imputation")
    zero_mask = np.abs(X) <= zero_tol
    masked = X.copy()
    masked[zero_mask] = np.nan
    medians = np.nanmedian(masked, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    imputed = np.where(zero_mask, medians[np.newaxis, :], X)
    return imputed


def knn_impute(X: np.ndarray, n_neighbors: int = 5, zero_tol: float = 1e-8) -> np.ndarray:
    log(f"Running KNNImputer with k={n_neighbors}")
    zero_mask = np.abs(X) <= zero_tol
    X_nan = X.copy()
    X_nan[zero_mask] = np.nan
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_nan)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed_scaled = imputer.fit_transform(X_scaled)
    imputed = scaler.inverse_transform(imputed_scaled)
    imputed[~zero_mask] = X[~zero_mask]
    return imputed


def svd_impute(X: np.ndarray, n_components: int = 50, zero_tol: float = 1e-8) -> np.ndarray:
    log(f"Running TruncatedSVD imputation with {n_components} components")
    zero_mask = np.abs(X) <= zero_tol
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(X)
    reconstructed = reduced @ svd.components_
    imputed = X.copy()
    imputed[zero_mask] = reconstructed[zero_mask]
    return imputed


def log1p_cpm_dense(matrix: np.ndarray) -> np.ndarray:
    libsize = np.sum(matrix, axis=1)
    scale = 1e4 / (libsize + 1e-8)
    scaled = matrix * scale[:, None]
    return np.log1p(scaled)


def graph_smooth_impute(
    X: np.ndarray,
    n_neighbors: int = 30,
    n_components: int = 30,
    include_self: bool = True,
    seed: int = 42,
) -> np.ndarray:
    log(f"Running graph smoothing with k={n_neighbors}, pca={n_components}")
    n_cells = X.shape[0]
    if n_cells <= 1:
        return X.astype(np.float32)
    k = max(1, min(n_neighbors, n_cells - 1))
    pca_components = min(n_components, X.shape[1])

    norm = log1p_cpm_dense(X.astype(np.float32))
    pca = PCA(n_components=pca_components, random_state=seed, svd_solver="randomized")
    embedding = pca.fit_transform(norm)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embedding)
    graph = nn.kneighbors_graph(embedding, n_neighbors=k, mode="connectivity")
    if include_self:
        import scipy.sparse as sp

        graph = graph + sp.eye(graph.shape[0], format="csr")
    graph = normalize(graph, norm="l1", axis=1)
    smoothed = graph.dot(X.astype(np.float32))
    return np.asarray(smoothed, dtype=np.float32)


METHOD_REGISTRY = {
    "gene_median": gene_median_impute,
    "knn_impute": knn_impute,
    "svd_impute": svd_impute,
    "graph_smooth": graph_smooth_impute,
}


def save_matrix(matrix: np.ndarray, method: str, dataset_id: str, disease: str, tissue: str, output_root: Path) -> Path:
    dest = output_root / method / dataset_id / disease
    dest.mkdir(parents=True, exist_ok=True)
    file_path = dest / f"{tissue}.npy"
    log(f"Saving {method} imputation to {file_path}")
    np.save(file_path, matrix.astype(np.float32))
    return file_path


def infer_dataset_id(path: Path) -> str:
    dataset_id = path.stem
    return dataset_id


def main():
    parser = argparse.ArgumentParser(description="Run lightweight imputation baselines for new datasets.")
    parser.add_argument("--input_file", required=True, help="Path to the h5ad file.")
    parser.add_argument("--methods", type=str, default="gene_median,knn_impute", help="Comma-separated imputation methods to run.")
    parser.add_argument("--output_root", type=str, default="output", help="Root directory to store imputed arrays.")
    parser.add_argument("--dataset_id", type=str, default=None, help="Optional dataset identifier (defaults to h5ad stem).")
    parser.add_argument("--disease", required=True, help="Disease/condition label for directory naming.")
    parser.add_argument("--tissue", required=True, help="Tissue label for directory naming.")
    parser.add_argument("--knn_neighbors", type=int, default=5, help="Neighbors for KNN imputer.")
    parser.add_argument("--svd_components", type=int, default=50, help="Number of components for SVD imputer.")
    parser.add_argument("--graph_neighbors", type=int, default=30, help="Neighbors for graph smoothing.")
    parser.add_argument("--graph_pca_components", type=int, default=30, help="PCA components for graph smoothing.")
    parser.add_argument("--graph_include_self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero_tolerance", type=float, default=1e-8, help="Threshold for considering values as zero.")
    args = parser.parse_args()

    methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Available: {list(METHOD_REGISTRY.keys())}")

    adata = read_h5ad(args.input_file)
    log(f"Loaded AnnData with shape {adata.shape}")
    X = to_dense(adata.X).astype(np.float32)

    dataset_id = args.dataset_id or infer_dataset_id(Path(args.input_file))
    for method in methods:
        if method == "gene_median":
            imputed = gene_median_impute(X, zero_tol=args.zero_tolerance)
        elif method == "knn_impute":
            imputed = knn_impute(X, n_neighbors=args.knn_neighbors, zero_tol=args.zero_tolerance)
        elif method == "svd_impute":
            imputed = svd_impute(X, n_components=args.svd_components, zero_tol=args.zero_tolerance)
        elif method == "graph_smooth":
            imputed = graph_smooth_impute(
                X,
                n_neighbors=args.graph_neighbors,
                n_components=args.graph_pca_components,
                include_self=args.graph_include_self,
                seed=42,
            )
        else:
            raise RuntimeError("Unhandled method")
        save_matrix(imputed, method, dataset_id, args.disease, args.tissue, Path(args.output_root))


if __name__ == "__main__":
    main()
