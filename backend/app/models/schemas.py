"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"


class SourceType(str, Enum):
    DOCUMENT = "document"
    WEB = "web"


class FeedbackRating(int, Enum):
    VERY_BAD = 1
    BAD = 2
    NEUTRAL = 3
    GOOD = 4
    EXCELLENT = 5


# =============================================================================
# Document Schemas
# =============================================================================

class DocumentBase(BaseModel):
    """Base document schema."""
    filename: str
    doc_type: DocumentType = DocumentType.PDF
    language: str = "fr"


class DocumentCreate(DocumentBase):
    """Schema for document creation."""
    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    original_filename: str
    file_size: int
    page_count: Optional[int] = None
    processed: bool = False
    upload_date: datetime
    doc_metadata: Dict[str, Any] = {}


class DocumentListResponse(BaseModel):
    """Schema for document list response."""
    documents: List[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    doc_id: UUID
    filename: str
    status: str
    chunks_created: int
    page_count: int
    processing_time_ms: int


# =============================================================================
# Chunk Schemas
# =============================================================================

class ChunkBase(BaseModel):
    """Base chunk schema."""
    content: str
    page_number: Optional[int] = None
    chunk_index: int


class ChunkResponse(ChunkBase):
    """Schema for chunk response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    document_id: UUID
    token_count: Optional[int] = None
    chunk_metadata: Dict[str, Any] = {}


# =============================================================================
# Source/Citation Schemas
# =============================================================================

class SourceBase(BaseModel):
    """Base source schema for citations."""
    type: SourceType
    title: str
    relevance_score: Optional[float] = None


class DocumentSource(SourceBase):
    """Document source citation."""
    type: SourceType = SourceType.DOCUMENT
    document_id: Optional[UUID] = None
    filename: str
    page: Optional[int] = None
    chunk_preview: Optional[str] = None


class WebSource(SourceBase):
    """Web source citation."""
    type: SourceType = SourceType.WEB
    url: str
    domain: str
    retrieved_date: datetime


Source = DocumentSource | WebSource


# =============================================================================
# Query Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Schema for query request."""
    question: str = Field(..., min_length=1, max_length=2000)
    use_web_search: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    language: Optional[str] = Field(default=None, pattern="^(fr|en|ar)$")
    include_sources: bool = True


class QueryResponse(BaseModel):
    """Schema for query response."""
    answer: str
    sources: List[Source] = []
    language: str
    web_search_used: bool = False
    processing_time_ms: int
    query_id: UUID


class StreamQueryResponse(BaseModel):
    """Schema for streaming query response chunk."""
    chunk: str
    done: bool = False
    sources: Optional[List[Source]] = None
    processing_time_ms: Optional[int] = None


# =============================================================================
# Training Schemas
# =============================================================================

class TrainingExampleCreate(BaseModel):
    """Schema for creating a training example."""
    question: str = Field(..., min_length=1)
    context: Optional[str] = None
    ideal_answer: str = Field(..., min_length=1)
    rating: Optional[FeedbackRating] = None


class TrainingExampleResponse(TrainingExampleCreate):
    """Schema for training example response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    used_count: int = 0
    created_at: datetime
    updated_at: datetime


class TrainingBatchRequest(BaseModel):
    """Schema for batch training examples."""
    examples: List[TrainingExampleCreate]


class TrainingBatchResponse(BaseModel):
    """Schema for batch training response."""
    created_count: int
    examples: List[TrainingExampleResponse]


# =============================================================================
# Feedback Schemas
# =============================================================================

class FeedbackRequest(BaseModel):
    """Schema for feedback submission."""
    query_id: UUID
    rating: FeedbackRating
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""
    success: bool
    message: str


# =============================================================================
# Health Check Schemas
# =============================================================================

class ServiceHealth(BaseModel):
    """Health status of a service."""
    name: str
    status: str  # "healthy" | "unhealthy" | "degraded"
    latency_ms: Optional[int] = None
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str  # "healthy" | "unhealthy" | "degraded"
    timestamp: datetime
    services: List[ServiceHealth]
    version: str = "1.0.0"


# =============================================================================
# Statistics Schemas
# =============================================================================

class StatsResponse(BaseModel):
    """Schema for statistics response."""
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_response_time_ms: float
    languages: Dict[str, int]
    top_sources: List[Dict[str, Any]]
