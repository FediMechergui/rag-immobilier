"""
Celery tasks for training and fine-tuning.
Handles training example creation, feedback processing, and model improvements.
"""
from uuid import UUID
from typing import Dict, Any, List
from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.celery_app import celery_app
from app.models.database import TrainingExample, QueryHistory

settings = get_settings()

# Create sync engine for Celery tasks
sync_engine = create_engine(settings.sync_database_url)
SyncSessionLocal = sessionmaker(bind=sync_engine)


@celery_app.task(bind=True, name="app.tasks.training_tasks.process_feedback")
def process_feedback_task(
    self,
    query_id: str,
    rating: int,
    comment: str = None
) -> Dict[str, Any]:
    """
    Process user feedback and potentially create training examples.
    
    High-rated responses (4-5) are automatically converted to training examples.
    Low-rated responses (1-2) are flagged for review.
    
    Args:
        query_id: UUID of the query
        rating: User rating (1-5)
        comment: Optional user comment
        
    Returns:
        Processing result
    """
    db = SyncSessionLocal()
    try:
        # Get the query history
        query = db.query(QueryHistory).filter(QueryHistory.id == UUID(query_id)).first()
        if not query:
            return {"success": False, "error": "Query not found"}
        
        # Update query with feedback
        query.feedback_rating = rating
        query.feedback_comment = comment
        db.commit()
        
        # For high ratings, create training example
        if rating >= 4:
            # Check if training example already exists
            existing = db.query(TrainingExample).filter(
                TrainingExample.question == query.question
            ).first()
            
            if not existing:
                training_example = TrainingExample(
                    question=query.question,
                    context=None,  # Will be filled if we have sources
                    ideal_answer=query.answer,
                    language=query.sources[0].get("language", "fr") if query.sources else "fr",
                    rating=rating,
                    used_count=0,
                    example_metadata={
                        "source_query_id": str(query_id),
                        "auto_generated": True,
                        "user_rating": rating
                    }
                )
                db.add(training_example)
                db.commit()
                
                return {
                    "success": True,
                    "training_example_created": True,
                    "query_id": query_id
                }
        
        return {
            "success": True,
            "training_example_created": False,
            "query_id": query_id
        }
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.training_tasks.create_training_example")
def create_training_example_task(
    self,
    question: str,
    ideal_answer: str,
    context: str = None,
    language: str = "fr",
    tags: List[str] = None
) -> Dict[str, Any]:
    """
    Create a new training example manually.
    
    Args:
        question: The question
        ideal_answer: The ideal answer
        context: Optional context that was used
        language: Language code
        tags: Optional tags for categorization
        
    Returns:
        Creation result
    """
    db = SyncSessionLocal()
    try:
        training_example = TrainingExample(
            question=question,
            context=context,
            ideal_answer=ideal_answer,
            language=language,
            rating=5,  # Manual examples are considered high quality
            used_count=0,
            example_metadata={
                "manual": True,
                "tags": tags or []
            }
        )
        db.add(training_example)
        db.commit()
        
        return {
            "success": True,
            "training_example_id": str(training_example.id)
        }
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(name="app.tasks.training_tasks.update_example_usage")
def update_example_usage_task(example_ids: List[str]) -> Dict[str, Any]:
    """
    Update usage count for training examples that were used in responses.
    
    Args:
        example_ids: List of training example UUIDs
        
    Returns:
        Update result
    """
    db = SyncSessionLocal()
    try:
        updated = 0
        for example_id in example_ids:
            example = db.query(TrainingExample).filter(
                TrainingExample.id == UUID(example_id)
            ).first()
            if example:
                example.used_count = (example.used_count or 0) + 1
                updated += 1
        
        db.commit()
        return {"success": True, "updated": updated}
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(name="app.tasks.training_tasks.generate_synthetic_examples")
def generate_synthetic_examples_task(
    document_id: str = None,
    count: int = 5
) -> Dict[str, Any]:
    """
    Generate synthetic training examples from document content.
    Uses the LLM to create question-answer pairs from chunks.
    
    Args:
        document_id: Optional specific document to use
        count: Number of examples to generate
        
    Returns:
        Generation result
    """
    from app.core.ollama_client import get_ollama_client
    from app.models.database import Chunk, Document
    
    db = SyncSessionLocal()
    try:
        ollama = get_ollama_client()
        
        # Get chunks to generate examples from
        query = db.query(Chunk).join(Document)
        if document_id:
            query = query.filter(Chunk.document_id == UUID(document_id))
        
        # Select diverse chunks (skip very short ones)
        chunks = query.filter(Chunk.token_count > 50).order_by(
            Chunk.id  # deterministic ordering
        ).limit(count * 2).all()
        
        if not chunks:
            return {"success": False, "error": "No suitable chunks found"}
        
        generated = 0
        for chunk in chunks[:count]:
            # Build a prompt asking the LLM to generate a Q&A pair
            generation_prompt = f"""Tu es un expert en immobilier français. À partir de l'extrait suivant, génère UNE question pertinente et sa réponse idéale.

EXTRAIT:
{chunk.content[:1500]}

Réponds EXACTEMENT dans ce format JSON (rien d'autre):
{{"question": "...", "answer": "..."}}

JSON:"""
            
            try:
                raw = ollama.generate(generation_prompt)
                # Try to parse the JSON from the response
                import json as _json
                import re
                # Find JSON object in the response
                match = re.search(r'\{[^{}]*"question"\s*:\s*"[^"]+"\s*,\s*"answer"\s*:\s*"[^"]+"\s*\}', raw, re.DOTALL)
                if not match:
                    # Try a more lenient parse
                    match = re.search(r'\{.*?\}', raw, re.DOTALL)
                
                if match:
                    parsed = _json.loads(match.group())
                    question = parsed.get("question", "").strip()
                    answer = parsed.get("answer", "").strip()
                    
                    if question and answer and len(question) > 10 and len(answer) > 20:
                        # Detect language from the chunk
                        from langdetect import detect
                        try:
                            lang = detect(chunk.content[:500])
                            lang = lang if lang in ("fr", "en", "ar") else "fr"
                        except Exception:
                            lang = "fr"
                        
                        example = TrainingExample(
                            question=question,
                            context=chunk.content[:500],
                            ideal_answer=answer,
                            language=lang,
                            rating=3,  # synthetic = moderate quality
                            used_count=0,
                            example_metadata={
                                "synthetic": True,
                                "source_document_id": str(chunk.document_id),
                                "source_chunk_id": str(chunk.id),
                            }
                        )
                        db.add(example)
                        generated += 1
            except Exception as e:
                logger.warning(f"Failed to generate synthetic example: {e}")
                continue
        
        db.commit()
        return {"success": True, "generated": generated}
    
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(name="app.tasks.training_tasks.cleanup_low_quality_examples")
def cleanup_low_quality_examples_task() -> Dict[str, Any]:
    """
    Remove or flag low-quality training examples.
    Examples with low ratings or no usage may be removed.
    
    Returns:
        Cleanup result
    """
    db = SyncSessionLocal()
    try:
        # Delete examples with rating < 3 and no manual flag
        deleted = db.query(TrainingExample).filter(
            TrainingExample.rating < 3,
            TrainingExample.example_metadata["manual"].astext != "true"
        ).delete(synchronize_session=False)
        
        db.commit()
        return {"success": True, "deleted": deleted}
        
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}
    
    finally:
        db.close()
