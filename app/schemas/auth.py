from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)
    client_secret: str = Field(min_length=1, max_length=256)
    # Space-delimited subset of the client's scopes. Omitted -> all of them.
    scope: str | None = Field(default=None, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str
