"""
Documents API Endpoints
Handles document upload, listing, and deletion.
Supports both sync (FastAPI background tasks) and async (Celery) processing.
"""
import os
import shutil
import time
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
import structlog

from app.core.config import get_settings
from app.models.database import get_db, Document, Chunk
from app.models.schemas import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse
)
from app.services.pdf_processor import get_pdf_processor
from app.services.embeddings import get_embedding_service

logger = structlog.get_logger()
settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])


# Try to import Celery tasks (may fail if RabbitMQ not available)
try:
    from app.tasks.document_tasks import process_document_task, delete_document_task, reindex_document_task
    CELERY_AVAILABLE = True
except Exception:
    CELERY_AVAILABLE = False
    logger.warning("Celery tasks not available, using FastAPI background tasks")


async def process_document_background(
    doc_id: UUID,
    file_path: str,
    original_filename: str
):
    """Background task to process uploaded document (non-Celery fallback)."""
    from app.models.database import AsyncSessionLocal
    
    pdf_processor = get_pdf_processor()
    embedding_service = get_embedding_service()
    
    async with AsyncSessionLocal() as db:
        try:
            # Update status to processing
            result = await db.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.doc_metadata = {**(doc.doc_metadata or {}), "status": "processing"}
                await db.commit()
            
            # Process PDF
            chunks, doc_metadata = pdf_processor.process_pdf(file_path, original_filename)
            
            if not chunks:
                logger.warning("No chunks extracted", doc_id=str(doc_id))
                if doc:
                    doc.doc_metadata = {**(doc.doc_metadata or {}), "status": "error", "error": "No text extracted"}
                    await db.commit()
                return
            
            # Update status
            if doc:
                doc.doc_metadata = {**(doc.doc_metadata or {}), "status": "generating_embeddings"}
                await db.commit()
            
            # Generate embeddings in batches
            chunk_texts = [c.content for c in chunks]
            embeddings = embedding_service.embed_texts(chunk_texts)
            
            # Create chunk records
            for chunk, embedding in zip(chunks, embeddings):
                chunk_record = Chunk(
                    document_id=doc_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    embedding=embedding,
                    token_count=chunk.token_count,
                    chunk_metadata=chunk.metadata
                )
                db.add(chunk_record)
            
            # Update document as processed
            result = await db.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if doc:
                doc.processed = True
                doc.page_count = doc_metadata.get("page_count", 0)
                doc.language = doc_metadata.get("language", "fr")
                doc.doc_metadata = {
                    **(doc_metadata or {}),
                    "status": "ready",
                    "chunk_count": len(chunks),
                    "total_tokens": sum(c.token_count for c in chunks)
                }
            
            await db.commit()
            logger.info(
                "Document processed successfully",
                doc_id=str(doc_id),
                chunks=len(chunks)
            )
            
        except Exception as e:
            logger.error("Document processing failed", doc_id=str(doc_id), error=str(e))
            await db.rollback()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_celery: bool = Query(default=True, description="Use Celery for background processing"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a PDF document for processing.
    
    The document will be processed asynchronously using either:
    - Celery workers (if available and use_celery=True)
    - FastAPI background tasks (fallback)
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate file size
    content = await file.read()
    file_size = len(content)
    max_size = settings.max_upload_size_mb * 1024 * 1024
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB"
        )
    
    # Generate unique filename
    doc_id = uuid4()
    safe_filename = f"{doc_id}.pdf"
    file_path = os.path.join(settings.upload_directory, safe_filename)
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_directory, exist_ok=True)
    
    # Save file
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error("Failed to save file", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Validate PDF
    pdf_processor = get_pdf_processor()
    is_valid, error = pdf_processor.validate_pdf(file_path)
    
    if not is_valid:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=error)
    
    # Create document record
    document = Document(
        id=doc_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_size=file_size,
        processed=False,
        doc_metadata={"status": "queued"}
    )
    db.add(document)
    await db.commit()
    
    # Schedule processing
    task_id = None
    if use_celery and CELERY_AVAILABLE:
        try:
            # Use Celery for background processing
            task = process_document_task.delay(
                str(doc_id),
                file_path,
                safe_filename,
                file.filename,
                "fr"  # Default language
            )
            task_id = task.id
            logger.info("Document queued for Celery processing", doc_id=str(doc_id), task_id=task_id)
        except Exception as e:
            logger.warning(f"Celery task failed, falling back to background task: {e}")
            background_tasks.add_task(
                process_document_background,
                doc_id,
                file_path,
                file.filename
            )
    else:
        # Use FastAPI background tasks
        background_tasks.add_task(
            process_document_background,
            doc_id,
            file_path,
            file.filename
        )
    
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="processing",
        chunks_created=0,  # Will be updated after processing
        page_count=0,
        processing_time_ms=processing_time_ms
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all uploaded documents.
    """
    # Get total count
    count_result = await db.execute(select(func.count(Document.id)))
    total = count_result.scalar()
    
    # Get documents
    result = await db.execute(
        select(Document)
        .order_by(Document.upload_date.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific document by ID.
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse.model_validate(document)


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document and all associated chunks.
    """
    # Check if document exists
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    file_path = os.path.join(settings.upload_directory, document.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete document (cascades to chunks)
    await db.execute(delete(Document).where(Document.id == doc_id))
    await db.commit()
    
    return {"message": "Document deleted successfully", "doc_id": str(doc_id)}


@router.get("/{doc_id}/status")
async def get_document_status(
    doc_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing status of a document.
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_id)
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get chunk count
    chunk_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)
    )
    chunk_count = chunk_result.scalar()
    
    return {
        "doc_id": str(doc_id),
        "filename": document.original_filename,
        "processed": document.processed,
        "page_count": document.page_count,
        "chunk_count": chunk_count,
        "language": document.language
    }
