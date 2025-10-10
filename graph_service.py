from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Literal
import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering

# UMAP is optional; fall back to PCA if unavailable
try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def _pairwise_cosine_from_normalized(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for already L2-normalized embeddings."""
    # Cosine equals dot product when rows are unit vectors
    return X @ X.T

def _knn_indices_cosine(X: np.ndarray, k: int) -> np.ndarray:
    """Return for each row the indices of top-k nearest neighbors by cosine similarity (excluding self)."""
    n = X.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.int32)

    if _HAS_FAISS:
        # Use inner product (dot) for normalized vectors = cosine
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X.astype(np.float32, copy=False))
        # Query itself then drop self (first index)
        sims, idx = index.search(X.astype(np.float32, copy=False), k + 1)
        return idx[:, 1:]
    else:
        # Pure NumPy fallback
        S = _pairwise_cosine_from_normalized(X)
        # Exclude self by setting diagonal to -inf
        np.fill_diagonal(S, -np.inf)
        # Argpartition to get k largest per row
        idx = np.argpartition(-S, kth=np.minimum(k, n-1)-1, axis=1)[:, :k]
        # Optional: sort top-k by similarity descending for nicer ordering
        row_indices = np.arange(n)[:, None]
        row_sims = S[row_indices, idx]
        order = np.argsort(-row_sims, axis=1)
        idx_sorted = idx[row_indices, order]
        return idx_sorted

def reduce_dimensions(
    X: np.ndarray,
    method: Literal["none", "pca", "umap", "tsne"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    if X.size == 0 or n_components <= 0 or method == "none":
        return None

    if method == "pca" or (method == "umap" and not _HAS_UMAP):
        # PCA is fast and deterministic
        n_components = min(n_components, X.shape[1])
        coords = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
        return coords.astype(np.float32, copy=False)

    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(X)
        return coords.astype(np.float32, copy=False)

    if method == "tsne":
        # t-SNE is slower; use perplexity rule-of-thumb
        perplexity = max(5, min(30, X.shape[0] // 3))
        tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
        coords = tsne.fit_transform(X)
        return coords.astype(np.float32, copy=False)

    # Fallback
    return None

def cluster_embeddings(
    X: np.ndarray,
    method: Literal["none", "kmeans", "agglomerative"] = "none",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    if method == "none" or X.size == 0:
        return None

    if n_clusters is None:
        # Heuristic: ~sqrt(n/2), min 2, max 20
        n = X.shape[0]
        n_clusters = max(2, min(20, int(np.sqrt(max(2, n/2)))))

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        return labels.astype(np.int32, copy=False)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels.astype(np.int32, copy=False)

    return None

def build_similarity_graph(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    include_embeddings: bool = False,
    dr_method: Literal["none", "pca", "umap", "tsne"] = "pca",
    n_components: int = 2,
    cluster_method: Literal["none", "kmeans", "agglomerative"] = "none",
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-friendly graph dict from embeddings and options."""
    X = embeddings.astype(np.float32, copy=False)
    n = X.shape[0]
    if n == 0:
        return {"nodes": [], "edges": []}

    # Dimensionality reduction (for x/y layout)
    coords = reduce_dimensions(X, method=dr_method, n_components=n_components)
    # Clustering
    clusters = cluster_embeddings(X, method=cluster_method, n_clusters=n_clusters)

    # Nodes
    nodes = []
    for i in range(n):
        node = {
            "id": str(i),
            "label": labels[i] if labels and i < len(labels) else None,
        }
        if node["label"] is None:
            # default: trimmed preview of the text; caller may pass labels for better display
            node["label"] = f"Doc {i}"
        if include_embeddings:
            node["embedding"] = X[i].tolist()
        if coords is not None and coords.shape[1] >= 2:
            node["x"], node["y"] = float(coords[i, 0]), float(coords[i, 1])
        if clusters is not None:
            node["cluster"] = int(clusters[i])
        nodes.append(node)

    # Edges
    edges = []
    added = set()  # to avoid duplicates for kNN case
    if threshold is not None:
        S = _pairwise_cosine_from_normalized(X)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j])
                if w >= float(threshold):
                    edges.append({"source": str(i), "target": str(j), "weight": round(w, 6)})
    else:
        k = int(top_k) if top_k is not None else 5
        k = max(1, min(k, max(1, n - 1)))
        idx = _knn_indices_cosine(X, k=k)
        # Build undirected edges; add once per pair
        S = _pairwise_cosine_from_normalized(X)
        for i in range(n):
            for j in idx[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a == b:
                    continue
                key = (a, b)
                if key in added:
                    continue
                added.add(key)
                w = float(S[a, b])
                edges.append({"source": str(a), "target": str(b), "weight": round(w, 6)})

    return {"nodes": nodes, "edges": edges}
