"""Health check router — used by Docker healthchecks and load balancers."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is running."""
    return {"status": "ok"}
