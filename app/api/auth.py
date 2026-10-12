"""POST /auth/token: trade a client id + secret for a short-lived access token."""

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import enforce_rate_limit
from app.core.auth import create_access_token, verify_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.auth import TokenRequest, TokenResponse

# Rate limited by IP (there is no token yet): this is the one endpoint where
# someone can guess secrets, so it needs the limiter most.
router = APIRouter(
    prefix="/auth", tags=["auth"], dependencies=[Depends(enforce_rate_limit)]
)
logger = get_logger(__name__)


@router.post("/token", response_model=TokenResponse)
async def issue_token(body: TokenRequest) -> TokenResponse:
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Authentication is disabled.")

    client = verify_client(settings, body.client_id, body.client_secret)
    if client is None:
        # Same answer for "no such client" and "wrong secret".
        logger.info("token request rejected", client_id=body.client_id)
        raise HTTPException(
            status_code=401,
            detail="Invalid client credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    allowed = set(client.scopes)
    requested = set(body.scope.split()) if body.scope else allowed
    if not requested or not requested <= allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Requested scope not permitted. This client may use: "
            f"{' '.join(sorted(allowed))}",
        )

    token, expires_in = create_access_token(
        settings, subject=body.client_id, scopes=sorted(requested)
    )
    logger.info("token issued", client_id=body.client_id, scope=sorted(requested))
    return TokenResponse(
        access_token=token, expires_in=expires_in, scope=" ".join(sorted(requested))
    )
