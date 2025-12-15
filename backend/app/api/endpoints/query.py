"""
Query API Endpoints
Handles RAG queries and streaming responses.
"""
import time
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import structlog
import json

from app.core.config import get_settings
from app.models.database import get_db, QueryHistory
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    FeedbackResponse
)
from app.services.rag_pipeline import get_rag_pipeline
from app.services.web_search import get_web_search_service

logger = structlog.get_logger()
settings = get_settings()
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Query the RAG pipeline with a question.
    
    - **question**: The question to ask
    - **use_web_search**: Whether to include web search results
    - **top_k**: Number of document chunks to retrieve
    - **language**: Response language (fr, en, ar) or auto-detect
    """
    rag_pipeline = get_rag_pipeline()
    web_sources = None
    
    # Perform web search if enabled
    if request.use_web_search and settings.web_search_enabled:
        try:
            web_search_service = get_web_search_service()
            web_sources = await web_search_service.search(db, request.question)
        except Exception as e:
            logger.warning("Web search failed", error=str(e))
    
    # Execute RAG query
    try:
        response = await rag_pipeline.query(
            db=db,
            question=request.question,
            language=request.language,
            top_k=request.top_k,
            web_sources=web_sources
        )
        return response
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Stream the RAG response for real-time output.
    
    Returns Server-Sent Events (SSE) with response chunks.
    """
    rag_pipeline = get_rag_pipeline()
    web_sources = None
    
    # Perform web search if enabled
    if request.use_web_search and settings.web_search_enabled:
        try:
            web_search_service = get_web_search_service()
            web_sources = await web_search_service.search(db, request.question)
        except Exception as e:
            logger.warning("Web search failed", error=str(e))
    
    async def generate_stream():
        """Generate SSE stream."""
        try:
            # Get retrieval results first
            chunks_with_scores = await rag_pipeline.retrieve_chunks(
                db, request.question, request.top_k
            )
            context, doc_sources = rag_pipeline.format_context(chunks_with_scores)
            
            # Debug: Log retrieved chunks
            logger.info(
                "Retrieved chunks for streaming",
                question=request.question[:50],
                num_chunks=len(chunks_with_scores),
                sources=[{
                    "filename": s.filename,
                    "page": s.page,
                    "score": s.relevance_score
                } for s in doc_sources[:3]]  # Log top 3
            )
            
            # Build prompt
            language = request.language or rag_pipeline.detect_language(request.question)
            prompt = rag_pipeline.build_prompt(request.question, context, language)
            
            # Debug: Log prompt length
            logger.info("Generated prompt", length=len(prompt), preview=prompt[:200])
            
            # Stream response
            start_time = time.time()
            full_response = ""
            
            async for chunk in rag_pipeline.ollama_client.stream_generate(prompt):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
            
            # Debug: Log full response
            logger.info("Stream completed", response_length=len(full_response), preview=full_response[:200])
            
            # Send final message with sources
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            all_sources = list(doc_sources)
            if web_sources:
                all_sources.extend(web_sources)
            
            # Convert sources to JSON-serializable format (UUIDs to strings)
            sources_data = []
            for s in all_sources:
                source_dict = s.model_dump()
                # Convert UUID to string if present
                if 'document_id' in source_dict and source_dict['document_id'] is not None:
                    source_dict['document_id'] = str(source_dict['document_id'])
                # Convert datetime to ISO string if present
                if 'retrieved_date' in source_dict and source_dict['retrieved_date'] is not None:
                    source_dict['retrieved_date'] = source_dict['retrieved_date'].isoformat()
                sources_data.append(source_dict)
            
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'sources': sources_data, 'processing_time_ms': processing_time_ms})}\n\n"
            
        except Exception as e:
            logger.error("Stream generation failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit feedback for a query response.
    
    - **query_id**: The ID of the query to provide feedback for
    - **rating**: Rating from 1 (very bad) to 5 (excellent)
    - **comment**: Optional feedback comment
    """
    # Find query
    result = await db.execute(
        select(QueryHistory).where(QueryHistory.id == request.query_id)
    )
    query_history = result.scalar_one_or_none()
    
    if not query_history:
        raise HTTPException(status_code=404, detail="Query not found")
    
    # Update feedback
    await db.execute(
        update(QueryHistory)
        .where(QueryHistory.id == request.query_id)
        .values(
            feedback_rating=request.rating.value,
            feedback_comment=request.comment
        )
    )
    await db.commit()
    
    logger.info(
        "Feedback submitted",
        query_id=str(request.query_id),
        rating=request.rating.value
    )
    
    return FeedbackResponse(
        success=True,
        message="Feedback submitted successfully"
    )


@router.get("/history")
async def get_query_history(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Get query history for analytics.
    """
    result = await db.execute(
        select(QueryHistory)
        .order_by(QueryHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    queries = result.scalars().all()
    
    return {
        "queries": [
            {
                "id": str(q.id),
                "question": q.question,
                "answer_preview": q.answer[:200] + "..." if len(q.answer) > 200 else q.answer,
                "web_search_used": q.web_search_used,
                "processing_time_ms": q.processing_time_ms,
                "feedback_rating": q.feedback_rating,
                "created_at": q.created_at.isoformat()
            }
            for q in queries
        ],
        "total": len(queries)
    }
