"""
Immobilier RAG Pipeline - FastAPI Application
Main entry point for the backend API.
"""
from contextlib import asynccontextmanager
from datetime import datetime
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import httpx

from app.core.config import get_settings
from app.core.ollama_client import get_ollama_client
from app.api import api_router
from app.models.schemas import HealthResponse, ServiceHealth

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("=== Starting Immobilier RAG Pipeline ===", flush=True)
    logger.info("Starting Immobilier RAG Pipeline")
    
    # Initialize services on startup
    try:
        print("Loading embedding service...", flush=True)
        # Pre-load embedding model
        from app.services.embeddings import get_embedding_service
        embedding_service = get_embedding_service()
        if embedding_service:
            print("Embedding service initialized", flush=True)
            logger.info("Embedding service initialized")
    except Exception as e:
        print(f"Failed to initialize embedding service: {e}", flush=True)
        logger.error("Failed to initialize embedding service", error=str(e))
    
    try:
        print("Checking Ollama connection...", flush=True)
        # Check Ollama connection
        ollama = get_ollama_client()
        health = await ollama.check_health()
        if health["status"] == "healthy":
            print(f"Ollama connection established: {settings.ollama_model}", flush=True)
            logger.info("Ollama connection established", model=settings.ollama_model)
        else:
            print(f"Ollama not available: {health}", flush=True)
            logger.warning("Ollama not available", details=health)
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}", flush=True)
        logger.error("Failed to connect to Ollama", error=str(e))
    
    print("=== Startup complete, yielding ===", flush=True)
    yield
    
    print("=== Shutting down ===", flush=True)
    logger.info("Shutting down Immobilier RAG Pipeline")


# Create FastAPI application
app = FastAPI(
    title="Immobilier RAG Pipeline",
    description="""
    🏠 **Real Estate RAG Assistant API**
    
    A production-ready RAG (Retrieval-Augmented Generation) pipeline specialized 
    for the French real estate (immobilier) domain.
    
    ## Features
    
    - 📄 **PDF Document Processing**: Upload and process real estate documents
    - 🔍 **Semantic Search**: Find relevant information using AI embeddings
    - 💬 **Intelligent Q&A**: Get accurate answers with source citations
    - 🌐 **Web Search**: Optional web search from trusted real estate sources
    - 📚 **Training**: Improve responses with custom examples
    
    ## Supported Languages
    
    - 🇫🇷 French (default)
    - 🇬🇧 English
    - 🇸🇦 Arabic
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time_ms=int(process_time * 1000)
    )
    
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include API routes
app.include_router(api_router, prefix="/api")


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Comprehensive health check for all services.
    """
    services = []
    overall_status = "healthy"
    
    # Check PostgreSQL
    try:
        from app.models.database import AsyncSessionLocal
        from sqlalchemy import text
        
        start = time.time()
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        latency = int((time.time() - start) * 1000)
        
        services.append(ServiceHealth(
            name="postgresql",
            status="healthy",
            latency_ms=latency,
            details={"host": settings.postgres_host}
        ))
    except Exception as e:
        services.append(ServiceHealth(
            name="postgresql",
            status="unhealthy",
            details={"error": str(e)}
        ))
        overall_status = "unhealthy"
    
    # Check ChromaDB (optional — not used for primary vector search, pgvector handles that)
    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=3.0) as client:
            try:
                response = await client.get(f"{settings.chroma_url}/api/v2/heartbeat")
                response.raise_for_status()
            except Exception:
                response = await client.get(f"{settings.chroma_url}/api/v1/heartbeat")
                response.raise_for_status()
        latency = int((time.time() - start) * 1000)
        
        services.append(ServiceHealth(
            name="chromadb",
            status="healthy",
            latency_ms=latency,
            details={"host": settings.chroma_host, "note": "optional — not used for primary vector search"}
        ))
    except Exception as e:
        services.append(ServiceHealth(
            name="chromadb",
            status="degraded",
            details={"error": str(e), "note": "optional service — pgvector handles vector search"}
        ))
        # ChromaDB is optional; don't degrade or fail overall health
    
    # Check Ollama
    try:
        ollama = get_ollama_client()
        start = time.time()
        health = await ollama.check_health()
        latency = int((time.time() - start) * 1000)
        
        services.append(ServiceHealth(
            name="ollama",
            status=health["status"],
            latency_ms=latency,
            details={
                "model": settings.ollama_model,
                "model_available": health.get("model_available", False)
            }
        ))
        if health["status"] != "healthy":
            overall_status = "degraded" if overall_status == "healthy" else overall_status
    except Exception as e:
        services.append(ServiceHealth(
            name="ollama",
            status="unhealthy",
            details={"error": str(e)}
        ))
        overall_status = "degraded" if overall_status == "healthy" else overall_status
    
    # Check Redis (optional)
    try:
        import redis.asyncio as redis
        start = time.time()
        r = redis.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        latency = int((time.time() - start) * 1000)
        
        services.append(ServiceHealth(
            name="redis",
            status="healthy",
            latency_ms=latency,
            details={"host": settings.redis_host}
        ))
    except Exception as e:
        services.append(ServiceHealth(
            name="redis",
            status="degraded",
            details={"error": str(e), "note": "Optional service"}
        ))
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services,
        version="1.0.0"
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Immobilier RAG Pipeline",
        "version": "1.0.0",
        "description": "Real Estate RAG Assistant API",
        "docs": "/docs",
        "health": "/health"
    }


# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
