from pydantic import BaseModel, Field

from app.schemas.query import RetrievedChunk


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk] = []
