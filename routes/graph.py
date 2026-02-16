from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from auth import get_current_user
from embedding_service import get_embedding_service
from graph_service import build_similarity_graph
from models import User
from schemas.graph import GraphRequest, GraphResponse

router = APIRouter(tags=["graph"])


@router.post("/api/graph", response_model=GraphResponse)
def graph(req: GraphRequest, current_user: User = Depends(get_current_user)) -> GraphResponse:
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    threshold = req.threshold
    top_k = None if threshold is not None else req.top_k

    service = get_embedding_service()
    X = service.encode(req.documents)

    graph_dict = build_similarity_graph(
        embeddings=X,
        labels=req.labels,
        threshold=threshold,
        top_k=top_k,
        include_embeddings=req.include_embeddings,
        dr_method=req.dr_method,
        n_components=req.n_components,
        cluster_method=req.cluster,
        n_clusters=req.n_clusters,
    )
    return GraphResponse(**graph_dict)
