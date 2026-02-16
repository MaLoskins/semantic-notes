from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GraphRequest(BaseModel):
    documents: List[str] = Field(..., description="List of texts to graph")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum cosine similarity for an edge")
    top_k: Optional[int] = Field(5, ge=1, description="kNN edges per node if threshold not provided")
    include_embeddings: bool = Field(False, description="Include embedding vectors in node payload")
    dr_method: Literal["none", "pca", "umap", "tsne"] = Field("pca", description="Dimensionality reduction for x/y")
    n_components: int = Field(2, ge=2, le=10, description="Output dimensions for DR")
    cluster: Literal["none", "kmeans", "agglomerative"] = Field("none", description="Clustering algorithm")
    n_clusters: Optional[int] = Field(None, ge=2, description="Number of clusters (optional)")
    labels: Optional[List[str]] = Field(None, description="Optional labels per document")


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
