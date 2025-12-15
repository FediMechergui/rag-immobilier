"""
Celery tasks for document processing.
Handles PDF upload, text extraction, chunking, and embedding generation in the background.
"""
import os
import asyncio
from uuid import UUID
from typing import Dict, Any, Optional
from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.celery_app import celery_app
from app.services.pdf_processor import get_pdf_processor
from app.services.embeddings import get_embedding_service
from app.models.database import Document, Chunk, Base

settings = get_settings()

# Create sync engine for Celery tasks
sync_engine = create_engine(settings.sync_database_url)
SyncSessionLocal = sessionmaker(bind=sync_engine)


def get_sync_db():
    """Get synchronous database session for Celery tasks."""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.document_tasks.process_document")
def process_document_task(
    self,
    document_id: str,
    file_path: str,
    filename: str,
    original_filename: str,
    language: str = "fr"
) -> Dict[str, Any]:
    """
    Process a document in the background.
    
    This task:
    1. Extracts text from PDF
    2. Creates chunks
    3. Generates embeddings
    4. Stores everything in the database
    
    Args:
        document_id: UUID of the document
        file_path: Path to the uploaded file
        filename: Stored filename
        original_filename: Original filename
        language: Document language
        
    Returns:
        Processing result with metadata
    """
    db = SyncSessionLocal()
    try:
        # Update document status to processing
        doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
        if not doc:
            return {"success": False, "error": "Document not found"}
        
        doc.doc_metadata = {**doc.doc_metadata, "status": "processing"}
        db.commit()
        
        # Get services
        pdf_processor = get_pdf_processor()
        embedding_service = get_embedding_service()
        
        # Process PDF
        chunks_data, doc_metadata = pdf_processor.process_pdf(file_path, original_filename)
        
        if not chunks_data:
            doc.doc_metadata = {**doc.doc_metadata, "status": "error", "error": "No text extracted"}
            db.commit()
            return {"success": False, "error": "No text extracted from PDF"}
        
        # Update document metadata
        doc.page_count = doc_metadata.get("page_count", 0)
        doc.processed = False  # Will be True after embeddings
        doc.doc_metadata = {
            **doc.doc_metadata,
            **doc_metadata,
            "status": "generating_embeddings"
        }
        db.commit()
        
        # Generate embeddings and create chunks
        chunk_count = 0
        for chunk_data in chunks_data:
            # Generate embedding
            embedding = embedding_service.embed_text(chunk_data.content)
            
            # Create chunk record
            chunk = Chunk(
                document_id=UUID(document_id),
                chunk_index=chunk_data.chunk_index,
                content=chunk_data.content,
                page_number=chunk_data.page_number,
                token_count=chunk_data.token_count,
                embedding=embedding,
                chunk_metadata=chunk_data.metadata
            )
            db.add(chunk)
            chunk_count += 1
            
            # Update progress periodically
            if chunk_count % 10 == 0:
                self.update_state(
                    state="PROGRESS",
                    meta={"current": chunk_count, "total": len(chunks_data)}
                )
                db.commit()
        
        # Final commit
        db.commit()
        
        # Update document as processed
        doc.processed = True
        doc.doc_metadata = {
            **doc.doc_metadata,
            "status": "ready",
            "chunk_count": chunk_count,
            "total_tokens": sum(c.token_count for c in chunks_data)
        }
        db.commit()
        
        return {
            "success": True,
            "document_id": document_id,
            "chunk_count": chunk_count,
            "page_count": doc.page_count,
            "language": doc_metadata.get("language", "unknown")
        }
        
    except Exception as e:
        # Update document with error status
        try:
            doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
            if doc:
                doc.doc_metadata = {**doc.doc_metadata, "status": "error", "error": str(e)}
                db.commit()
        except:
            pass
        
        # Re-raise for Celery retry
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.document_tasks.delete_document")
def delete_document_task(self, document_id: str) -> Dict[str, Any]:
    """
    Delete a document and its chunks in the background.
    
    Args:
        document_id: UUID of the document to delete
        
    Returns:
        Deletion result
    """
    db = SyncSessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
        if not doc:
            return {"success": False, "error": "Document not found"}
        
        # Delete file if exists
        file_path = os.path.join(settings.upload_directory, doc.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete chunks (cascade should handle this, but be explicit)
        db.query(Chunk).filter(Chunk.document_id == UUID(document_id)).delete()
        
        # Delete document
        db.delete(doc)
        db.commit()
        
        return {"success": True, "document_id": document_id}
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.document_tasks.reindex_document")
def reindex_document_task(self, document_id: str) -> Dict[str, Any]:
    """
    Re-generate embeddings for a document.
    Useful when changing embedding model or fixing issues.
    
    Args:
        document_id: UUID of the document to reindex
        
    Returns:
        Reindexing result
    """
    db = SyncSessionLocal()
    try:
        embedding_service = get_embedding_service()
        
        # Get all chunks for document
        chunks = db.query(Chunk).filter(Chunk.document_id == UUID(document_id)).all()
        
        if not chunks:
            return {"success": False, "error": "No chunks found"}
        
        # Update document status
        doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
        if doc:
            doc.doc_metadata = {**doc.doc_metadata, "status": "reindexing"}
            db.commit()
        
        # Regenerate embeddings
        updated = 0
        for chunk in chunks:
            embedding = embedding_service.embed_text(chunk.content)
            chunk.embedding = embedding
            updated += 1
            
            if updated % 10 == 0:
                self.update_state(
                    state="PROGRESS",
                    meta={"current": updated, "total": len(chunks)}
                )
                db.commit()
        
        db.commit()
        
        # Update document status
        if doc:
            doc.doc_metadata = {**doc.doc_metadata, "status": "ready"}
            db.commit()
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_updated": updated
        }
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(name="app.tasks.document_tasks.cleanup_orphaned_files")
def cleanup_orphaned_files_task() -> Dict[str, Any]:
    """
    Clean up orphaned files in the upload directory.
    Files that don't have corresponding database records will be deleted.
    
    Returns:
        Cleanup result
    """
    db = SyncSessionLocal()
    try:
        # Get all document filenames from database
        docs = db.query(Document.filename).all()
        db_filenames = {doc.filename for doc in docs}
        
        # Get all files in upload directory
        upload_dir = settings.upload_directory
        if not os.path.exists(upload_dir):
            return {"success": True, "deleted": 0}
        
        deleted = 0
        for filename in os.listdir(upload_dir):
            if filename not in db_filenames:
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted += 1
        
        return {"success": True, "deleted": deleted}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()
