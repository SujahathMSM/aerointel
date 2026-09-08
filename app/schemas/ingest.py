from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    id: int
    dimensions: int
