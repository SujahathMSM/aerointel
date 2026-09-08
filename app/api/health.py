from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.api.deps import SessionDep

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live():
    """Process is up. Checks nothing — cheap, always succeeds if alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(session: SessionDep):
    """Can we serve traffic? Verifies the database is reachable."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any failure means "not ready"
        raise HTTPException(status_code=503, detail="database not reachable")
    return {"status": "ready"}
