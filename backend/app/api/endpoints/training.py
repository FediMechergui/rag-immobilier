"""
Training API Endpoints
Handles training examples and few-shot learning management.
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
import structlog

from app.models.database import get_db, TrainingExample
from app.models.schemas import (
    TrainingExampleCreate,
    TrainingExampleResponse,
    TrainingBatchRequest,
    TrainingBatchResponse
)

logger = structlog.get_logger()
router = APIRouter(prefix="/training", tags=["training"])


@router.post("/examples", response_model=TrainingExampleResponse)
async def create_training_example(
    request: TrainingExampleCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a single training example for few-shot learning.
    
    Training examples help improve the quality of responses by providing
    the model with high-quality question-answer pairs.
    """
    example = TrainingExample(
        question=request.question,
        context=request.context,
        ideal_answer=request.ideal_answer,
        rating=request.rating.value if request.rating else None
    )
    db.add(example)
    await db.commit()
    await db.refresh(example)
    
    logger.info("Training example created", example_id=str(example.id))
    
    return TrainingExampleResponse.model_validate(example)


@router.post("/examples/batch", response_model=TrainingBatchResponse)
async def create_training_examples_batch(
    request: TrainingBatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create multiple training examples at once.
    """
    created_examples = []
    
    for ex_data in request.examples:
        example = TrainingExample(
            question=ex_data.question,
            context=ex_data.context,
            ideal_answer=ex_data.ideal_answer,
            rating=ex_data.rating.value if ex_data.rating else None
        )
        db.add(example)
        created_examples.append(example)
    
    await db.commit()
    
    # Refresh all examples
    for ex in created_examples:
        await db.refresh(ex)
    
    logger.info("Batch training examples created", count=len(created_examples))
    
    return TrainingBatchResponse(
        created_count=len(created_examples),
        examples=[TrainingExampleResponse.model_validate(ex) for ex in created_examples]
    )


@router.get("/examples", response_model=List[TrainingExampleResponse])
async def list_training_examples(
    skip: int = 0,
    limit: int = 100,
    min_rating: int = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all training examples.
    
    - **min_rating**: Filter by minimum rating (1-5)
    """
    query = select(TrainingExample)
    
    if min_rating is not None:
        query = query.where(TrainingExample.rating >= min_rating)
    
    query = query.order_by(
        TrainingExample.rating.desc().nullslast(),
        TrainingExample.used_count.desc()
    ).offset(skip).limit(limit)
    
    result = await db.execute(query)
    examples = result.scalars().all()
    
    return [TrainingExampleResponse.model_validate(ex) for ex in examples]


@router.get("/examples/{example_id}", response_model=TrainingExampleResponse)
async def get_training_example(
    example_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific training example.
    """
    result = await db.execute(
        select(TrainingExample).where(TrainingExample.id == example_id)
    )
    example = result.scalar_one_or_none()
    
    if not example:
        raise HTTPException(status_code=404, detail="Training example not found")
    
    return TrainingExampleResponse.model_validate(example)


@router.put("/examples/{example_id}", response_model=TrainingExampleResponse)
async def update_training_example(
    example_id: UUID,
    request: TrainingExampleCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update a training example.
    """
    result = await db.execute(
        select(TrainingExample).where(TrainingExample.id == example_id)
    )
    example = result.scalar_one_or_none()
    
    if not example:
        raise HTTPException(status_code=404, detail="Training example not found")
    
    # Update fields
    example.question = request.question
    example.context = request.context
    example.ideal_answer = request.ideal_answer
    if request.rating:
        example.rating = request.rating.value
    
    await db.commit()
    await db.refresh(example)
    
    return TrainingExampleResponse.model_validate(example)


@router.delete("/examples/{example_id}")
async def delete_training_example(
    example_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a training example.
    """
    result = await db.execute(
        select(TrainingExample).where(TrainingExample.id == example_id)
    )
    example = result.scalar_one_or_none()
    
    if not example:
        raise HTTPException(status_code=404, detail="Training example not found")
    
    await db.execute(
        delete(TrainingExample).where(TrainingExample.id == example_id)
    )
    await db.commit()
    
    return {"message": "Training example deleted", "example_id": str(example_id)}


@router.get("/stats")
async def get_training_stats(
    db: AsyncSession = Depends(get_db)
):
    """
    Get training data statistics.
    """
    # Total count
    total_result = await db.execute(
        select(func.count(TrainingExample.id))
    )
    total = total_result.scalar()
    
    # Count by rating
    rating_result = await db.execute(
        select(
            TrainingExample.rating,
            func.count(TrainingExample.id)
        ).group_by(TrainingExample.rating)
    )
    ratings = {row[0]: row[1] for row in rating_result.fetchall()}
    
    # Average usage
    usage_result = await db.execute(
        select(func.avg(TrainingExample.used_count))
    )
    avg_usage = usage_result.scalar() or 0
    
    return {
        "total_examples": total,
        "by_rating": ratings,
        "average_usage": float(avg_usage),
        "high_quality_count": ratings.get(5, 0) + ratings.get(4, 0)
    }
