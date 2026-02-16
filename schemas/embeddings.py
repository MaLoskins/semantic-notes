from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    documents: List[str] = Field(..., description="List of texts to embed")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
