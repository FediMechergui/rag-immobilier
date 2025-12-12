from fastapi import APIRouter

from app.api.endpoints import documents, query, training

api_router = APIRouter()

api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(training.router)
