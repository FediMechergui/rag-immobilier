"""
RAG Pipeline Service
Orchestrates retrieval and generation for question answering.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from uuid import UUID, uuid4
import structlog
from langdetect import detect

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import get_settings
from app.core.ollama_client import get_ollama_client, OllamaClient
from app.core.prompts import SYSTEM_PROMPTS, DEFAULT_LANGUAGE, FEW_SHOT_EXAMPLES
from app.models.database import Chunk, Document, QueryHistory, TrainingExample
from app.models.schemas import DocumentSource, WebSource, Source, QueryResponse
from app.services.embeddings import get_embedding_service, EmbeddingService

logger = structlog.get_logger()
settings = get_settings()


class RAGPipeline:
    """RAG Pipeline for retrieval-augmented generation."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        ollama_client: Optional[OllamaClient] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_service: Embedding service instance
            ollama_client: Ollama client instance
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.ollama_client = ollama_client or get_ollama_client()
        self.top_k = settings.top_k
        self.similarity_threshold = settings.similarity_threshold
        
        logger.info("RAG Pipeline initialized")
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            lang = detect(text)
            return lang if lang in ["fr", "en", "ar"] else DEFAULT_LANGUAGE
        except Exception:
            return DEFAULT_LANGUAGE
    
    async def retrieve_chunks(
        self,
        db: AsyncSession,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks using vector similarity search.
        
        Args:
            db: Database session
            query: User query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (Chunk, similarity_score) tuples
        """
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        print(f"[RAG] Query: {query[:50]}...")
        print(f"[RAG] Embedding dims: {len(query_embedding)}, threshold: {threshold}")
        
        # Format embedding as PostgreSQL array format
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Vector similarity search using pgvector
        # Using cosine distance (1 - cosine_similarity)
        # Using bindparam for proper parameter handling with asyncpg
        from sqlalchemy import bindparam
        
        sql = text("""
            SELECT 
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.page_number,
                c.token_count,
                c.chunk_metadata,
                d.filename,
                d.original_filename,
                1 - (c.embedding <=> CAST(:embedding AS vector)) as similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """).bindparams(
            bindparam('embedding', value=embedding_str),
            bindparam('limit', value=top_k * 2)
        )
        
        result = await db.execute(sql)
        rows = result.fetchall()
        
        print(f"[RAG] SQL returned {len(rows)} rows")
        if rows:
            print(f"[RAG] Top 3 similarities: {[round(r.similarity, 3) for r in rows[:3]]}")
        
        # Filter by threshold and limit
        chunks_with_scores = []
        for row in rows:
            if row.similarity >= threshold:
                # Create Chunk object from row
                chunk = Chunk(
                    id=row.id,
                    document_id=row.document_id,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    page_number=row.page_number,
                    token_count=row.token_count,
                    chunk_metadata={
                        **row.chunk_metadata,
                        "filename": row.filename,
                        "original_filename": row.original_filename
                    }
                )
                chunks_with_scores.append((chunk, row.similarity))
        
        # Limit to top_k
        chunks_with_scores = chunks_with_scores[:top_k]
        
        print(f"[RAG] After threshold filter: {len(chunks_with_scores)} chunks")
        
        logger.info(
            "Retrieved chunks",
            query_preview=query[:50],
            retrieved=len(chunks_with_scores),
            top_similarity=chunks_with_scores[0][1] if chunks_with_scores else 0
        )
        
        return chunks_with_scores
    
    def format_context(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[DocumentSource]]:
        """
        Format retrieved chunks into context string.
        DO NOT include source markers - sources are handled separately.
        
        Args:
            chunks_with_scores: List of (Chunk, score) tuples
            
        Returns:
            Tuple of (context_string, list of sources)
        """
        if not chunks_with_scores:
            return "Aucun document pertinent trouvé dans la base de connaissances.", []
        
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            filename = chunk.chunk_metadata.get("original_filename", chunk.chunk_metadata.get("filename", "Unknown"))
            page = chunk.page_number
            
            # Format context WITHOUT source markers - just the content
            # Number each excerpt for clarity
            context_parts.append(f"--- Extrait {i} ---\n{chunk.content}")
            
            # Create source object (displayed separately in frontend)
            sources.append(DocumentSource(
                type="document",
                title=filename,
                document_id=chunk.document_id,
                filename=filename,
                page=page,
                relevance_score=score,
                chunk_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            ))
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sources
    
    async def get_few_shot_examples(
        self,
        db: AsyncSession,
        limit: int = 2
    ) -> List[Dict[str, str]]:
        """Get few-shot examples from database or defaults."""
        try:
            # Try to get from database
            result = await db.execute(
                select(TrainingExample)
                .where(TrainingExample.rating >= 4)
                .order_by(TrainingExample.used_count.desc())
                .limit(limit)
            )
            examples = result.scalars().all()
            
            if examples:
                return [
                    {
                        "question": ex.question,
                        "context": ex.context or "",
                        "answer": ex.ideal_answer
                    }
                    for ex in examples
                ]
        except Exception:
            pass
        
        # Fall back to default examples
        return FEW_SHOT_EXAMPLES[:limit]
    
    def build_prompt(
        self,
        question: str,
        context: str,
        language: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build the full prompt with context and examples.
        
        Args:
            question: User question
            context: Retrieved context
            language: Response language
            few_shot_examples: Optional few-shot examples
            
        Returns:
            Complete prompt string
        """
        # Get system prompt for language
        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[DEFAULT_LANGUAGE])
        
        # Build few-shot section if examples provided
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = "\n\nEXEMPLES:\n"
            for ex in few_shot_examples:
                few_shot_section += f"""
Question: {ex['question']}
Contexte: {ex.get('context', 'N/A')[:300]}...
Réponse: {ex['answer'][:500]}...

---
"""
        
        # Build final prompt
        prompt = system_prompt.format(context=context, question=question)
        
        if few_shot_section:
            # Insert few-shot examples before the question
            prompt = prompt.replace(
                f"QUESTION: {question}",
                f"{few_shot_section}\nQUESTION: {question}"
            )
        
        return prompt
    
    async def generate_answer(
        self,
        prompt: str,
        stream: bool = False
    ) -> str:
        """
        Generate answer using Ollama.
        
        Args:
            prompt: Full prompt with context
            stream: Whether to stream the response
            
        Returns:
            Generated answer
        """
        if stream:
            # Return generator for streaming
            return self.ollama_client.stream_generate(prompt)
        else:
            return await self.ollama_client.agenerate(prompt)
    
    async def query(
        self,
        db: AsyncSession,
        question: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
        web_sources: Optional[List[WebSource]] = None,
        use_few_shot: bool = True
    ) -> QueryResponse:
        """
        Execute full RAG query pipeline.
        
        Args:
            db: Database session
            question: User question
            language: Response language (auto-detected if not provided)
            top_k: Number of chunks to retrieve
            web_sources: Optional web search results to include
            use_few_shot: Whether to include few-shot examples
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        query_id = uuid4()
        
        # Detect language
        response_language = language or self.detect_language(question)
        
        logger.info(
            "Processing query",
            query_id=str(query_id),
            language=response_language,
            question_preview=question[:100]
        )
        
        # Retrieve relevant chunks
        chunks_with_scores = await self.retrieve_chunks(db, question, top_k)
        
        # Format context and get document sources
        context, doc_sources = self.format_context(chunks_with_scores)
        
        # Add web sources to context if provided
        all_sources: List[Source] = list(doc_sources)
        if web_sources:
            web_context_parts = []
            for ws in web_sources:
                web_context_parts.append(
                    f"[Web: {ws.domain}, {ws.retrieved_date.strftime('%Y-%m-%d')}]\n{ws.title}"
                )
                all_sources.append(ws)
            
            if web_context_parts:
                context += "\n\n---\nSOURCES WEB:\n" + "\n\n".join(web_context_parts)
        
        # Get few-shot examples
        few_shot = None
        if use_few_shot:
            few_shot = await self.get_few_shot_examples(db)
        
        # Build prompt
        prompt = self.build_prompt(question, context, response_language, few_shot)
        
        # Generate answer
        answer = await self.generate_answer(prompt)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Save to query history
        try:
            history = QueryHistory(
                id=query_id,
                question=question,
                answer=answer,
                sources=[s.model_dump() for s in all_sources],
                web_search_used=bool(web_sources),
                processing_time_ms=processing_time_ms
            )
            db.add(history)
            await db.commit()
        except Exception as e:
            logger.error("Failed to save query history", error=str(e))
        
        logger.info(
            "Query completed",
            query_id=str(query_id),
            processing_time_ms=processing_time_ms,
            sources_count=len(all_sources)
        )
        
        return QueryResponse(
            answer=answer,
            sources=all_sources,
            language=response_language,
            web_search_used=bool(web_sources),
            processing_time_ms=processing_time_ms,
            query_id=query_id
        )


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
