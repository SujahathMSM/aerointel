from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunk(BaseModel):
    id: int
    distance: float
    source: str | None = None
    content: str


class QueryResponse(BaseModel):
    question: str
    refused: bool
    answer: str | None = None
    refusal_message: str | None = None
    chunks: list[RetrievedChunk] = []
