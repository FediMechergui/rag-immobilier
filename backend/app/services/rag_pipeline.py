"""
RAG Pipeline Service — Modern Hybrid Retrieval-Augmented Generation.

Implements:
  • Hybrid search  – pgvector cosine + PostgreSQL full-text (BM25-style) with RRF fusion
  • Query rewriting – Ollama-based expansion for bilingual (FR/EN) recall
  • Cross-encoder re-ranking – LLM-based relevance scoring after retrieval
  • Bilingual awareness – seamless French/English context and generation
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import asyncio
from uuid import UUID, uuid4
import structlog
from langdetect import detect

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, bindparam

from app.core.config import get_settings
from app.core.ollama_client import get_ollama_client, OllamaClient
from app.core.prompts import (
    SYSTEM_PROMPTS, DEFAULT_LANGUAGE, FEW_SHOT_EXAMPLES,
    QUERY_REWRITE_PROMPT, RERANK_PROMPT,
)
from app.models.database import Chunk, Document, QueryHistory, TrainingExample
from app.models.schemas import DocumentSource, WebSource, Source, QueryResponse
from app.services.embeddings import get_embedding_service, EmbeddingService

logger = structlog.get_logger()
settings = get_settings()

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion constant (k=60 is the standard from the RRF paper)
# ---------------------------------------------------------------------------
RRF_K = 60


class RAGPipeline:
    """Modern RAG Pipeline with hybrid retrieval, rewriting, and re-ranking."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.ollama_client = ollama_client or get_ollama_client()
        self.top_k = settings.top_k
        self.similarity_threshold = settings.similarity_threshold
        logger.info("RAG Pipeline initialized (hybrid mode)")

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------
    def detect_language(self, text_str: str) -> str:
        """Detect language of text."""
        try:
            lang = detect(text_str)
            return lang if lang in ["fr", "en", "ar"] else DEFAULT_LANGUAGE
        except Exception:
            return DEFAULT_LANGUAGE

    # ------------------------------------------------------------------
    # 1. QUERY REWRITING — bilingual expansion
    # ------------------------------------------------------------------
    async def rewrite_query(self, query: str, language: str) -> List[str]:
        """
        Use the LLM to produce expanded / translated query variants.

        Returns a list of query strings (always includes the original).
        Bilingual: if the query is FR, also produce an EN variant and vice-versa.
        """
        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query, language=language)
            raw = await self.ollama_client.agenerate(prompt)
            # Parse numbered lines (1. … 2. … 3. …)
            variants: List[str] = []
            for line in raw.strip().splitlines():
                cleaned = line.strip().lstrip("0123456789.-) ").strip()
                if cleaned and len(cleaned) > 5:
                    variants.append(cleaned)
            # Always keep the original first
            if query not in variants:
                variants.insert(0, query)
            return variants[:4]  # cap at 4 variants
        except Exception as e:
            logger.warning("Query rewriting failed, using original", error=str(e))
            return [query]

    # ------------------------------------------------------------------
    # 2. VECTOR SEARCH (pgvector cosine)
    # ------------------------------------------------------------------
    async def _vector_search(
        self,
        db: AsyncSession,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Return list of (chunk_id, similarity) from pgvector cosine search."""
        query_embedding = self.embedding_service.embed_text(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = text("""
            SELECT
                c.id,
                1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM chunks c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """).bindparams(
            bindparam("embedding", value=embedding_str),
            bindparam("limit", value=limit),
        )
        result = await db.execute(sql)
        return [(str(r.id), float(r.similarity)) for r in result.fetchall()]

    # ------------------------------------------------------------------
    # 3. FULL-TEXT / BM25-STYLE KEYWORD SEARCH
    # ------------------------------------------------------------------
    async def _keyword_search(
        self,
        db: AsyncSession,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        Return list of (chunk_id, ts_rank) using PostgreSQL full-text search.
        Uses plainto_tsquery for robustness (handles non-boolean input).
        Searches in both French and English text-search configs.
        """
        sql = text("""
            SELECT
                c.id,
                GREATEST(
                    ts_rank_cd(c.content_tsv, plainto_tsquery('french',  :query)),
                    ts_rank_cd(c.content_tsv, plainto_tsquery('english', :query))
                ) AS rank
            FROM chunks c
            WHERE c.content_tsv IS NOT NULL
              AND (
                  c.content_tsv @@ plainto_tsquery('french',  :query)
                  OR c.content_tsv @@ plainto_tsquery('english', :query)
              )
            ORDER BY rank DESC
            LIMIT :limit
        """).bindparams(
            bindparam("query", value=query),
            bindparam("limit", value=limit),
        )
        result = await db.execute(sql)
        return [(str(r.id), float(r.rank)) for r in result.fetchall()]

    # ------------------------------------------------------------------
    # 4. RECIPROCAL RANK FUSION
    # ------------------------------------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        *ranked_lists: List[Tuple[str, float]],
        k: int = RRF_K,
    ) -> Dict[str, float]:
        """
        Merge multiple ranked lists via Reciprocal Rank Fusion.

        Each list is [(id, score)]. RRF score = Σ 1/(k + rank_i).
        """
        fused: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
        return fused

    # ------------------------------------------------------------------
    # 5. FETCH FULL CHUNK OBJECTS
    # ------------------------------------------------------------------
    async def _fetch_chunks_by_ids(
        self,
        db: AsyncSession,
        chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """Fetch full Chunk rows (with document info) by IDs."""
        if not chunk_ids:
            return {}

        # Positional parameters for IN clause
        placeholders = ", ".join(f":id_{i}" for i in range(len(chunk_ids)))
        params = {f"id_{i}": cid for i, cid in enumerate(chunk_ids)}

        sql = text(f"""
            SELECT
                c.id, c.document_id, c.chunk_index, c.content,
                c.page_number, c.token_count, c.chunk_metadata,
                d.filename, d.original_filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id::text IN ({placeholders})
        """).bindparams(**params)

        result = await db.execute(sql)
        rows = {str(r.id): r for r in result.fetchall()}
        return rows

    # ------------------------------------------------------------------
    # 6. RETRIEVE (orchestrates hybrid pipeline)
    # ------------------------------------------------------------------
    async def retrieve_chunks(
        self,
        db: AsyncSession,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Hybrid retrieval: vector + keyword search fused with RRF.

        Steps:
          1. Rewrite/expand query (bilingual)
          2. Run vector search for EACH query variant
          3. Run keyword search for EACH query variant
          4. Fuse all results with RRF
          5. Re-rank top candidates with LLM cross-encoder
          6. Return final top_k with scores
        """
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        fetch_limit = top_k * 3  # retrieve more for fusion

        language = self.detect_language(query)

        # --- Step 1: Rewrite ---
        query_variants = await self.rewrite_query(query, language)
        logger.info("Query variants", variants=query_variants)

        # --- Step 2+3: Run vector & keyword searches for all variants ---
        all_vector_results: List[List[Tuple[str, float]]] = []
        all_keyword_results: List[List[Tuple[str, float]]] = []

        for variant in query_variants:
            vec_res = await self._vector_search(db, variant, fetch_limit)
            kw_res = await self._keyword_search(db, variant, fetch_limit)
            all_vector_results.append(vec_res)
            all_keyword_results.append(kw_res)

        # --- Step 4: RRF fusion ---
        fused_scores = self._reciprocal_rank_fusion(
            *all_vector_results, *all_keyword_results
        )

        # Sort by fused score descending
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        # Keep generous set for re-ranking
        candidate_ids = ranked_ids[: top_k * 2]

        # --- Step 5: Fetch full chunk data ---
        rows = await self._fetch_chunks_by_ids(db, candidate_ids)

        # --- Step 6: LLM Re-ranking ---
        reranked = await self._rerank(query, candidate_ids, rows)

        # Build final result
        chunks_with_scores: List[Tuple[Chunk, float]] = []
        for cid, score in reranked:
            row = rows.get(cid)
            if not row:
                continue
            chunk = Chunk(
                id=row.id,
                document_id=row.document_id,
                chunk_index=row.chunk_index,
                content=row.content,
                page_number=row.page_number,
                token_count=row.token_count,
                chunk_metadata={
                    **(row.chunk_metadata or {}),
                    "filename": row.filename,
                    "original_filename": row.original_filename,
                },
            )
            chunks_with_scores.append((chunk, score))

        chunks_with_scores = chunks_with_scores[:top_k]

        logger.info(
            "Hybrid retrieval complete",
            query_preview=query[:50],
            variants=len(query_variants),
            vector_hits=sum(len(v) for v in all_vector_results),
            keyword_hits=sum(len(k) for k in all_keyword_results),
            fused_candidates=len(fused_scores),
            final=len(chunks_with_scores),
            top_score=chunks_with_scores[0][1] if chunks_with_scores else 0,
        )

        return chunks_with_scores

    # ------------------------------------------------------------------
    # 7. LLM RE-RANKING
    # ------------------------------------------------------------------
    async def _rerank(
        self,
        query: str,
        candidate_ids: List[str],
        rows: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """
        Use the LLM as a cross-encoder to re-rank candidate chunks.

        For each candidate, ask the model to rate relevance 0-10.
        Falls back to RRF order if LLM fails.
        """
        if not candidate_ids:
            return []

        # Build a batch prompt — score all candidates at once
        passages = []
        valid_ids = []
        for cid in candidate_ids:
            row = rows.get(cid)
            if row:
                # Truncate passage for prompt budget
                content = row.content[:400] if row.content else ""
                passages.append(content)
                valid_ids.append(cid)

        if not passages:
            return []

        # Build the prompt
        prompt = RERANK_PROMPT.format(
            query=query,
            passages="\n".join(
                f"[{i+1}] {p}" for i, p in enumerate(passages)
            ),
            count=len(passages),
        )

        try:
            raw = await self.ollama_client.agenerate(prompt)
            scores = self._parse_rerank_scores(raw, len(passages))
            ranked = sorted(
                zip(valid_ids, scores), key=lambda x: x[1], reverse=True
            )
            return ranked
        except Exception as e:
            logger.warning("LLM re-ranking failed, using RRF order", error=str(e))
            # Fallback: assign decreasing scores based on original order
            return [(cid, 1.0 - i * 0.01) for i, cid in enumerate(valid_ids)]

    @staticmethod
    def _parse_rerank_scores(raw: str, expected_count: int) -> List[float]:
        """Parse numbered score lines from LLM output (e.g. '1. 8\n2. 5')."""
        import re
        scores = []
        for line in raw.strip().splitlines():
            m = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*10)?$', line.strip())
            if m:
                score = float(m.group(1))
                if score > 10:
                    score = 10.0
                scores.append(score / 10.0)  # normalise to 0-1
        # Pad or truncate
        while len(scores) < expected_count:
            scores.append(0.5)
        return scores[:expected_count]
    
    # ------------------------------------------------------------------
    # FORMAT CONTEXT
    # ------------------------------------------------------------------
    def format_context(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]],
        language: str = "fr",
    ) -> Tuple[str, List[DocumentSource]]:
        """
        Format retrieved chunks into a context string for the LLM.
        Sources are tracked separately for the frontend.
        """
        no_result_msgs = {
            "fr": "Aucun document pertinent trouvé dans la base de connaissances.",
            "en": "No relevant documents found in the knowledge base.",
        }
        if not chunks_with_scores:
            return no_result_msgs.get(language, no_result_msgs["fr"]), []

        context_parts: List[str] = []
        sources: List[DocumentSource] = []

        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            filename = chunk.chunk_metadata.get(
                "original_filename",
                chunk.chunk_metadata.get("filename", "Unknown"),
            )
            page = chunk.page_number

            # Bilingual label
            label = "Excerpt" if language == "en" else "Extrait"
            context_parts.append(f"--- {label} {i} ---\n{chunk.content}")

            sources.append(
                DocumentSource(
                    type="document",
                    title=filename,
                    document_id=chunk.document_id,
                    filename=filename,
                    page=page,
                    relevance_score=score,
                    chunk_preview=(
                        chunk.content[:200] + "..."
                        if len(chunk.content) > 200
                        else chunk.content
                    ),
                )
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    # ------------------------------------------------------------------
    # FEW-SHOT EXAMPLES (language-aware)
    # ------------------------------------------------------------------
    async def get_few_shot_examples(
        self,
        db: AsyncSession,
        language: str = "fr",
        limit: int = 2,
    ) -> List[Dict[str, str]]:
        """Get few-shot examples from database, preferring matching language."""
        try:
            result = await db.execute(
                select(TrainingExample)
                .where(TrainingExample.rating >= 4)
                .where(TrainingExample.language.in_([language, "fr"]))
                .order_by(TrainingExample.used_count.desc())
                .limit(limit)
            )
            examples = result.scalars().all()

            if examples:
                return [
                    {
                        "question": ex.question,
                        "context": ex.context or "",
                        "answer": ex.ideal_answer,
                    }
                    for ex in examples
                ]
        except Exception:
            pass

        return FEW_SHOT_EXAMPLES[:limit]

    # ------------------------------------------------------------------
    # BUILD PROMPT
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        question: str,
        context: str,
        language: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build the full prompt with context and few-shot examples."""
        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS[DEFAULT_LANGUAGE])

        few_shot_section = ""
        if few_shot_examples:
            header = "EXAMPLES:" if language == "en" else "EXEMPLES :"
            few_shot_section = f"\n\n{header}\n"
            for ex in few_shot_examples:
                q_label = "Question" if language == "en" else "Question"
                c_label = "Context" if language == "en" else "Contexte"
                a_label = "Answer" if language == "en" else "Réponse"
                few_shot_section += (
                    f"\n{q_label}: {ex['question']}\n"
                    f"{c_label}: {ex.get('context', 'N/A')[:300]}...\n"
                    f"{a_label}: {ex['answer'][:500]}...\n\n---\n"
                )

        prompt = system_prompt.format(context=context, question=question)

        # Insert few-shot before the user question
        q_marker = f"QUESTION: {question}" if language != "en" else f"QUESTION: {question}"
        if few_shot_section and q_marker in prompt:
            prompt = prompt.replace(q_marker, f"{few_shot_section}\n{q_marker}")

        return prompt

    # ------------------------------------------------------------------
    # GENERATE ANSWER
    # ------------------------------------------------------------------
    async def generate_answer(
        self,
        prompt: str,
        stream: bool = False,
    ):
        """Generate answer using Ollama (sync or streaming)."""
        if stream:
            return self.ollama_client.stream_generate(prompt)
        return await self.ollama_client.agenerate(prompt)

    # ------------------------------------------------------------------
    # FULL QUERY PIPELINE
    # ------------------------------------------------------------------
    async def query(
        self,
        db: AsyncSession,
        question: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
        web_sources: Optional[List[WebSource]] = None,
        use_few_shot: bool = True,
    ) -> QueryResponse:
        """
        Execute the full modern RAG pipeline:
          1. Language detection
          2. Hybrid retrieval (vector + keyword + RRF + re-ranking)
          3. Context formatting
          4. Few-shot injection
          5. LLM generation
          6. History persistence
        """
        start_time = time.time()
        query_id = uuid4()

        response_language = language or self.detect_language(question)

        logger.info(
            "Processing query (modern RAG)",
            query_id=str(query_id),
            language=response_language,
            question_preview=question[:100],
        )

        # Hybrid retrieval (includes rewriting, vector, keyword, RRF, re-ranking)
        chunks_with_scores = await self.retrieve_chunks(db, question, top_k)

        # Format context
        context, doc_sources = self.format_context(
            chunks_with_scores, response_language
        )

        # Append web sources
        all_sources: List[Source] = list(doc_sources)
        if web_sources:
            web_parts: List[str] = []
            for ws in web_sources:
                web_parts.append(
                    f"[Web: {ws.domain}, {ws.retrieved_date.strftime('%Y-%m-%d')}]\n{ws.title}"
                )
                all_sources.append(ws)
            if web_parts:
                web_header = "WEB SOURCES:" if response_language == "en" else "SOURCES WEB :"
                context += f"\n\n---\n{web_header}\n" + "\n\n".join(web_parts)

        # Few-shot examples (language-aware)
        few_shot = None
        if use_few_shot:
            few_shot = await self.get_few_shot_examples(db, response_language)

        # Build prompt & generate
        prompt = self.build_prompt(question, context, response_language, few_shot)
        answer = await self.generate_answer(prompt)

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Persist history
        try:
            history = QueryHistory(
                id=query_id,
                question=question,
                answer=answer,
                sources=[s.model_dump() for s in all_sources],
                web_search_used=bool(web_sources),
                processing_time_ms=processing_time_ms,
            )
            db.add(history)
            await db.commit()
        except Exception as e:
            logger.error("Failed to save query history", error=str(e))

        logger.info(
            "Query completed (modern RAG)",
            query_id=str(query_id),
            processing_time_ms=processing_time_ms,
            sources_count=len(all_sources),
        )

        return QueryResponse(
            answer=answer,
            sources=all_sources,
            language=response_language,
            web_search_used=bool(web_sources),
            processing_time_ms=processing_time_ms,
            query_id=query_id,
        )


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
