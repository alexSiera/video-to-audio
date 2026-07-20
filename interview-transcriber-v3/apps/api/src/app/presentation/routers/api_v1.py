"""API v1 router — aggregates all versioned routers under /api/v1."""

from fastapi import APIRouter

from app.presentation.routers.health import router as health_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
# Future routers:
# api_v1_router.include_router(auth_router, prefix="/auth")
# api_v1_router.include_router(projects_router, prefix="/projects")
# api_v1_router.include_router(uploads_router, prefix="/uploads")
# api_v1_router.include_router(jobs_router, prefix="/jobs")
# api_v1_router.include_router(transcripts_router, prefix="/transcripts")
