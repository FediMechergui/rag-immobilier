# CHANGELOG — rag-immobilier

> **Bilingual Real-Estate RAG Pipeline — French & English**
> All changes documented from initial codebase audit through modern RAG overhaul.

---

## Table of Contents

- [v2.0.0 — Modern RAG Overhaul (Feb 2026)](#v200--modern-rag-overhaul-feb-2026)
  - [Bug Fixes](#bug-fixes)
  - [New Features & Enhancements](#new-features--enhancements)
  - [Infrastructure Changes](#infrastructure-changes)
  - [Frontend Changes](#frontend-changes)
- [Algorithms & Architecture](#algorithms--architecture)
  - [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
  - [Query Rewriting (Bilingual Expansion)](#query-rewriting-bilingual-expansion)
  - [LLM Cross-Encoder Re-Ranking](#llm-cross-encoder-re-ranking)
  - [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
  - [Bilingual Full-Text Search](#bilingual-full-text-search)
- [Current State](#current-state)
- [Known Limitations](#known-limitations)
- [Notes & Technical Decisions](#notes--technical-decisions)
- [Future Roadmap](#future-roadmap)

---

## v2.0.0 — Modern RAG Overhaul (Feb 2026)

A full modernisation of the RAG pipeline: from a basic single-vector-search system to a **hybrid retrieval pipeline** with bilingual awareness, query rewriting, and LLM-based re-ranking.

### Bug Fixes

#### 1. TrainingExample Missing Columns (`database.py`, `init.sql`)
- **Problem**: The `TrainingExample` ORM model and SQL schema were missing the `language` and `example_metadata` columns. Any attempt to create training examples would fail with a database error.
- **Fix**: Added `language: Mapped[str] = mapped_column(String(10), default="fr")` and `example_metadata: Mapped[dict] = mapped_column(JSON, default=dict)` to the `TrainingExample` class in `backend/app/models/database.py`.
- **SQL**: Added `language VARCHAR(10) DEFAULT 'fr'` and `example_metadata JSONB DEFAULT '{}'` to the `training_examples` table in `backend/init.sql`.

#### 2. Frontend Documents List Crash (`api.ts`)
- **Problem**: The backend `GET /api/documents/` returns a `{ documents: [...], total: N }` envelope, but the frontend expected a flat `Document[]` array. This caused the document list to either crash or render empty.
- **Fix**: Rewrote `getDocuments()` in `frontend/src/services/api.ts` to unwrap the envelope and map backend field names (`original_filename`, `doc_metadata.status`, `doc_metadata.chunk_count`, `upload_date`) to frontend `Document` types.

#### 3. Web Scraper Not Wired In (`query.py`)
- **Problem**: A sophisticated domain-specific web scraper (`web_scraper.py`) existed but was never imported — the query endpoints imported the naive `web_search.py` module instead.
- **Fix**: Changed the import in `backend/app/api/endpoints/query.py` from `from app.services.web_search import ...` to `from app.services.web_scraper import get_web_scraping_service`.

#### 4. ChromaDB False Health Dependency (`main.py`)
- **Problem**: ChromaDB was treated as a critical service in the health check. If ChromaDB went down, the entire system reported `unhealthy` — even though **pgvector** handles all vector search.
- **Fix**: Demoted ChromaDB to an optional service. Its failure now reports `degraded` status with the note `"optional service — pgvector handles vector search"` and does **not** affect the overall health status.

#### 5. Stub Synthetic Training Generation (`training_tasks.py`)
- **Problem**: `generate_synthetic_examples_task()` was an empty stub that always returned `"Not implemented"`.
- **Fix**: Fully implemented the task:
  - Queries diverse chunks (token_count > 50) from the database.
  - Sends each chunk to Ollama with a French-language generation prompt.
  - Parses JSON `{"question": "...", "answer": "..."}` from LLM output using regex.
  - Auto-detects language of each chunk (FR/EN/AR).
  - Stores synthetic examples with `example_metadata.synthetic = true`, source document/chunk IDs, and a synthetic quality rating of 3.
- **Also fixed**: 3 references to the non-existent `metadata=` keyword argument were corrected to `example_metadata=` across `process_feedback_task`, `create_training_example_task`, and `generate_synthetic_examples_task`.

---

### New Features & Enhancements

#### 6. Hybrid Search — Vector + Full-Text + RRF (`rag_pipeline.py`, `init.sql`)

**SQL Infrastructure** (`init.sql`):
- Added `content_tsv tsvector` column to the `chunks` table.
- Created a bilingual tsvector trigger (`chunks_tsv_trigger()`) that auto-populates `content_tsv` on INSERT/UPDATE:
  ```sql
  setweight(to_tsvector('french',  content), 'A')  -- French terms weighted higher
  || setweight(to_tsvector('english', content), 'B')
  ```
- Added GIN index (`idx_chunks_content_tsv`) for fast full-text matching.
- Backfill statement for existing rows.

**Python Implementation** (`rag_pipeline.py`):
- `_vector_search()` — pgvector cosine distance search via `<=>` operator.
- `_keyword_search()` — PostgreSQL `ts_rank_cd` against both `french` and `english` text-search configs with `plainto_tsquery` for robustness.
- `_reciprocal_rank_fusion()` — merges all ranked lists with the standard RRF formula: `score = Σ 1/(k + rank)` where `k=60`.
- `retrieve_chunks()` — orchestrates the full hybrid pipeline (rewrite → vector → keyword → RRF → re-rank).

#### 7. Query Rewriting & Bilingual Expansion (`rag_pipeline.py`, `prompts.py`)
- `rewrite_query()` method sends the user's query to Ollama with a specialised prompt that:
  - Always produces at least one French and one English variant.
  - Expands domain abbreviations (DPE → diagnostic de performance énergétique).
  - Caps at 4 query variants to limit latency.
  - Falls back to the original query if the LLM call fails.
- New `QUERY_REWRITE_PROMPT` in `prompts.py` with explicit bilingual expansion rules.

#### 8. LLM Cross-Encoder Re-Ranking (`rag_pipeline.py`, `prompts.py`)
- `_rerank()` method asks Ollama to score each candidate passage 0–10 for relevance.
- All candidates are scored in a single LLM call (batch prompt) for efficiency.
- `_parse_rerank_scores()` parses numbered score lines from LLM output, normalises to 0–1.
- Graceful fallback: if the LLM fails, original RRF order is preserved with decreasing dummy scores.
- New `RERANK_PROMPT` in `prompts.py` with strict output format instructions.

#### 9. Bilingual Semantic Prompts (`prompts.py`)
- Complete rewrite of all system prompts:
  - **French prompt**: Explicitly states the model is bilingual FR-EN, must translate English excerpts, respond only in French.
  - **English prompt**: Same bilingual awareness, responds only in English.
  - **Arabic prompt**: Supports Arabic responses (bonus language).
- All prompts include 8 strict grounding rules (no hallucination, no source leaking, structured output, no repetition).
- **Few-shot examples** now include both French and English examples (previously French only):
  - FR: Frais de notaire (notary fees)
  - EN: Rental yield calculation
  - FR: Rentabilité locative (rental profitability)
- Added `WEB_SEARCH_QUERY_PROMPT` for web search query optimization.

#### 10. Bilingual Context Formatting (`rag_pipeline.py`)
- `format_context()` now accepts a `language` parameter.
- Excerpt labels change based on language ("Extrait" for FR, "Excerpt" for EN).
- No-result messages are language-aware.
- Web source headers adapt ("SOURCES WEB" for FR, "WEB SOURCES" for EN).

#### 11. Language-Aware Few-Shot Retrieval (`rag_pipeline.py`)
- `get_few_shot_examples()` queries the database for training examples matching the detected language.
- Falls back to the hardcoded `FEW_SHOT_EXAMPLES` from `prompts.py` if no database examples exist.
- `build_prompt()` inserts few-shot examples with language-appropriate labels (Question/Contexte/Réponse vs Question/Context/Answer).

#### 12. Streaming Endpoint Bilingual Fix (`query.py`)
- The `/api/query/stream` endpoint now detects language **before** context formatting and passes it through the pipeline.
- Previously, language detection happened after the context was already built (using a hardcoded French format).

---

### Infrastructure Changes

#### 13. Docker Compose Modernisation (`docker-compose.yml`)
- Removed deprecated `version: '3.8'` (Docker Compose V2 no longer requires it).
- Changed default LLM model from `mistral` to `qwen2.5:0.5b` in both `backend` and `celery-worker` service definitions:
  ```yaml
  OLLAMA_MODEL=${OLLAMA_MODEL:-qwen2.5:0.5b}
  ```
- Rationale: `qwen2.5:0.5b` (397 MB) was already available on the system, vs `mistral` (4+ GB). Faster startup, lower resource usage.

#### 14. Default Model Configuration (`config.py`)
- Changed `ollama_model` default from `"mistral"` to `"qwen2.5:0.5b"` in Python settings.
- All three config layers now agree: `docker-compose.yml` → environment variable → Python default.

#### 15. Nginx Reverse Proxy Fix (`nginx.conf`)
- **Problem**: Nginx crashed on startup because it tried to resolve the `backend` hostname at configuration load time — before Docker DNS was ready.
- **Fix**: Added dynamic DNS resolution with a variable:
  ```nginx
  resolver 127.0.0.11 valid=10s;
  set $backend_upstream http://backend:8080;
  ```
  All `proxy_pass` directives now use `$backend_upstream` instead of hardcoded `http://backend:8080`, enabling runtime DNS resolution.

---

### Frontend Changes

#### 16. Document List API Compatibility (`api.ts`)
- `getDocuments()` now handles both response shapes:
  - `{ documents: [...], total: N }` envelope (current backend)
  - Flat `Document[]` array (forward-compatible)
- Field mapping from backend response:
  | Backend Field | Frontend Field |
  |---|---|
  | `original_filename` | `filename` |
  | `doc_metadata.status` | `status` |
  | `doc_metadata.chunk_count` | `chunk_count` |
  | `upload_date` | `created_at` |
  | `doc_metadata.error` | `error_message` |

---

## Algorithms & Architecture

### Hybrid Retrieval Pipeline

The modern RAG pipeline follows this sequence for every user query:

```
User Query
    │
    ▼
┌─────────────────────┐
│  1. Language Detect  │  (langdetect: fr/en/ar)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Query Rewriting │  Ollama generates 2-3 bilingual variants
│     (FR + EN)       │  (always includes original query)
└─────────┬───────────┘
          │
          ▼  (for each query variant)
┌─────────────────────────────────────────┐
│  3a. Vector Search     3b. Keyword      │
│      (pgvector)            Search       │
│      cosine <=>            (tsvector)   │
│      IVFFlat index         GIN index    │
│                            FR+EN configs│
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────┐
│  4. RRF Fusion      │  score = Σ 1/(60 + rank)
│     (k=60)          │  merges all vector + keyword lists
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. LLM Re-Ranking  │  Ollama scores each passage 0-10
│     (cross-encoder) │  normalised to 0.0–1.0
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Context Build   │  top_k chunks formatted with
│     + Few-Shot      │  bilingual labels + examples
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  7. LLM Generation  │  Ollama (qwen2.5:0.5b)
│     (grounded)      │  with stop sequences + repeat_penalty
└─────────┬───────────┘
          │
          ▼
      QueryResponse
      (answer + sources + metadata)
```

### Query Rewriting (Bilingual Expansion)

- **Purpose**: Maximise recall by searching in both French and English, even if the user wrote in only one language.
- **Method**: Ollama generates 2–3 alternative queries from the original, always including at least one French and one English variant.
- **Domain awareness**: Expands abbreviations (DPE, PLU, SCI, etc.) and adds real-estate domain terms.
- **Cap**: Maximum 4 query variants to bound latency.
- **Fallback**: If the LLM call fails, only the original query is used.

### LLM Cross-Encoder Re-Ranking

- **Purpose**: Improve precision after the high-recall fusion step.
- **Method**: All candidate passages (truncated to 400 chars each) are sent to Ollama in a single batch prompt. The model rates each passage 0–10 for relevance to the query.
- **Score normalisation**: Raw scores are divided by 10 to produce 0.0–1.0 range.
- **Robustness**: Regex parsing handles varied LLM output formats. Missing scores default to 0.5. Scores > 10 are clamped.
- **Fallback**: If re-ranking fails entirely, the RRF order is preserved.

### Reciprocal Rank Fusion (RRF)

- **Paper**: Cormack, Clarke & Butt (2009) — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **Formula**: `RRF_score(d) = Σ_{r ∈ ranked_lists} 1 / (k + rank_r(d))`
- **k parameter**: 60 (standard from the paper, balances high-ranked and mid-ranked results)
- **Input lists**: All vector search results + all keyword search results from every query variant
- **Output**: A single sorted list of chunk IDs with fused scores

### Bilingual Full-Text Search

- **PostgreSQL config**: Uses two text-search configurations simultaneously — `french` and `english`.
- **Weighting**: French tokens get weight `'A'` (highest), English tokens get weight `'B'`. This biases toward French content (the primary language of the knowledge base) while still matching English terms.
- **Trigger**: Auto-populates the `content_tsv` column on every `INSERT` or `UPDATE` to the chunks table.
- **Query**: Uses `plainto_tsquery` (not `to_tsquery`) for robustness against user input that contains special characters.
- **Ranking**: `ts_rank_cd` with cover density ranking, taking the `GREATEST` of French and English scores.

---

## Current State

### Services (8 containers)

| Service | Container | Port | Status |
|---|---|---|---|
| Ollama (LLM) | `immobilier-ollama` | 11434 | ✅ Healthy |
| PostgreSQL + pgvector | `immobilier-postgres` | 5432 | ✅ Healthy |
| ChromaDB | `immobilier-chromadb` | 8000 | ✅ (optional) |
| FastAPI Backend | `immobilier-backend` | 8080 | ✅ Healthy |
| React Frontend (Nginx) | `immobilier-frontend` | 3000 | ✅ Healthy |
| Redis | `immobilier-redis` | 6379 | ✅ Healthy |
| RabbitMQ | `immobilier-rabbitmq` | 5672 / 15672 | ✅ Healthy |
| Celery Worker | `immobilier-celery-worker` | — | ✅ Running |

### Access Points

| Endpoint | URL |
|---|---|
| Frontend (Chat UI) | http://localhost:3000 |
| Backend API | http://localhost:8080 |
| Swagger Docs | http://localhost:8080/docs |
| ReDoc | http://localhost:8080/redoc |
| Health Check | http://localhost:8080/health |
| RabbitMQ Management | http://localhost:15672 (raguser/ragpassword) |

### Models

| Model | Purpose | Size | Location |
|---|---|---|---|
| `qwen2.5:0.5b` | LLM generation, rewriting, re-ranking | 397 MB | Ollama |
| `paraphrase-multilingual-mpnet-base-v2` | Embeddings (768 dim) | ~1.1 GB | HuggingFace (cached) |

### Database Schema (5 tables)

| Table | Key Columns | Indexes |
|---|---|---|
| `documents` | id, filename, language, processed, doc_metadata | filename, upload_date |
| `chunks` | id, document_id, content, embedding(768), content_tsv | IVFFlat (cosine), GIN (tsvector), document_id |
| `training_examples` | id, question, ideal_answer, language, example_metadata | — |
| `query_history` | id, question, answer, sources, processing_time_ms | created_at |
| `web_search_cache` | id, query_hash, results, expires_at | query_hash (unique), expires_at |

### End-to-End Test Result

```
Question: "Quels sont les frais de notaire pour acheter un bien immobilier en France?"
Language: fr (auto-detected)
Answer:   ✅ French answer returned (notary fees explanation from model knowledge)
Sources:  [] (no documents uploaded yet)
Time:     27,758 ms
Pipeline: query rewriting → hybrid retrieval → LLM generation
```

---

## Known Limitations

### Model Limitations
1. **qwen2.5:0.5b is a small model** — Only 500M parameters. Answer quality is limited, especially for complex multi-step reasoning. Larger models (qwen2.5:7b, mistral, llama3) would significantly improve output quality but require more VRAM.
2. **Re-ranking quality** — LLM-based re-ranking with a 0.5B model may not consistently produce accurate relevance scores. A dedicated cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) would be more reliable.
3. **Embedding model is general-purpose** — `paraphrase-multilingual-mpnet-base-v2` is good for multilingual similarity but not domain-specialised for real estate. Fine-tuning or using a French-specific model could improve retrieval precision.

### Pipeline Limitations
4. **No document chunking overlap** — Chunks are created without overlapping windows, so context can be lost at chunk boundaries.
5. **No chunk-level metadata filtering** — The hybrid search does not yet support filtering by document type, date range, or metadata fields.
6. **No conversation history / multi-turn** — Each query is independent. The pipeline does not maintain chat context across turns.
7. **Streaming + re-ranking latency** — The full pipeline (rewrite → search → re-rank → generate) takes ~28 seconds on a cold start with the small model. Streaming helps perception but total time is high.
8. **No embedding cache** — Query embeddings are computed fresh each time. Frequently repeated queries could benefit from caching.

### Infrastructure Limitations
9. **ChromaDB is redundant** — pgvector handles all vector operations. ChromaDB is included but unused; it adds container overhead.
10. **No TLS / HTTPS** — All services run over HTTP. Production deployment requires TLS termination.
11. **No authentication** — No API keys, JWT, or user management. All endpoints are publicly accessible.
12. **CORS is fully open** — `allow_origins=["*"]` is set for development. Must be restricted for production.
13. **IVFFlat index requires training data** — The IVFFlat index with `lists = 100` is created before any data exists. It needs to be rebuilt after a significant number of vectors are inserted for optimal performance.
14. **GPU required for Ollama** — Docker Compose reserves NVIDIA GPU. Systems without a compatible GPU need the GPU section removed from `docker-compose.yml`.

### Data Limitations
15. **No documents uploaded yet** — The knowledge base is empty. The RAG currently falls back to model knowledge only.
16. **No evaluation dataset** — There is no ground-truth Q&A dataset for measuring retrieval quality (recall@k, MRR) or answer quality (BLEU, ROUGE, human eval).
17. **Web scraper domain list is limited** — `web_scraper.py` targets specific French real estate sites; other valuable sources may be missing.

---

## Notes & Technical Decisions

### Why pgvector over ChromaDB?
- pgvector lives in the same PostgreSQL instance as the relational data, enabling JOINs between chunks and documents without cross-service calls.
- Full-text search (tsvector) is native to PostgreSQL — no separate search engine needed.
- ChromaDB remains in the stack for potential future use (metadata-filtered collections) but is not on the critical path.

### Why Reciprocal Rank Fusion?
- RRF is simple, parameter-free (aside from k), and has been shown to outperform individual rankers and even some learned fusion methods.
- It gracefully handles score incompatibility between vector similarity (0–1 cosine) and BM25-style ranking (unbounded ts_rank_cd) by working purely on rank positions.

### Why LLM Re-Ranking instead of a dedicated cross-encoder?
- Avoids adding a separate model to the stack (memory savings on constrained hardware).
- The same Ollama instance is reused.
- Trade-off: lower quality than a specialised cross-encoder, but zero additional infrastructure.

### Why `paraphrase-multilingual-mpnet-base-v2`?
- Supports 50+ languages including French and English in a single model.
- 768-dimensional embeddings — good balance of quality and dimension size.
- Normalised embeddings — cosine distance is equivalent to dot product.

### Why `qwen2.5:0.5b`?
- Already available on the development machine (no extra download).
- 397 MB vs 4+ GB for Mistral — faster cold starts.
- Sufficient for development and testing. Intended to be upgraded for production.

### Bilingual Design Philosophy
- The system **understands** both French and English documents but **responds** in the user's detected language.
- Query rewriting always produces variants in both languages to maximise recall across mixed-language corpora.
- tsvector weighting biases toward French (weight A) since the primary corpus is French real estate documents, but English terms are still indexed (weight B).

### Training Example Pipeline
- **Manual creation**: Via API endpoint → Celery task → stored with `manual: true` metadata.
- **Auto-generation from feedback**: Queries rated 4–5 stars are automatically converted to training examples.
- **Synthetic generation**: LLM generates Q&A pairs from document chunks, stored with `synthetic: true` metadata and a conservative quality rating of 3.
- **Quality cleanup**: Periodic task removes examples with rating < 3 that are not flagged as manual.

---

## Files Modified (Complete List)

| File | Type | Summary |
|---|---|---|
| `backend/app/services/rag_pipeline.py` | **Rewritten** | ~611 lines. Full modern RAG pipeline with hybrid search, query rewriting, RRF, LLM re-ranking, bilingual context, few-shot injection. |
| `backend/app/core/prompts.py` | **Rewritten** | ~190 lines. Bilingual system prompts (FR/EN/AR), QUERY_REWRITE_PROMPT, RERANK_PROMPT, bilingual few-shot examples. |
| `backend/app/api/endpoints/query.py` | Modified | Switched import from `web_search` to `web_scraper`. Fixed streaming endpoint language handling. |
| `backend/app/models/database.py` | Modified | Added `language` and `example_metadata` columns to `TrainingExample`. |
| `backend/app/tasks/training_tasks.py` | Modified | Fixed `metadata=` → `example_metadata=` (×3). Fully implemented `generate_synthetic_examples_task`. |
| `backend/app/main.py` | Modified | ChromaDB health check demoted to optional. |
| `backend/app/core/config.py` | Modified | Default model changed from `mistral` to `qwen2.5:0.5b`. |
| `backend/init.sql` | Modified | Added `language`, `example_metadata` to training_examples. Added `content_tsv`, GIN index, bilingual trigger to chunks. |
| `docker-compose.yml` | Modified | Removed `version: '3.8'`. Changed `OLLAMA_MODEL` default to `qwen2.5:0.5b` (backend + celery-worker). |
| `frontend/src/services/api.ts` | Modified | `getDocuments()` unwraps `{documents, total}` envelope, maps field names. |
| `frontend/nginx.conf` | Modified | Added runtime DNS resolver for backend upstream. |

---

## Future Roadmap

### Short-Term (Next Sprint)
- [ ] **Upload PDF documents** — Populate the knowledge base with French real estate documents (loi, contrats, guides).
- [ ] **Rebuild IVFFlat index** — After uploading documents, run `REINDEX INDEX idx_chunks_embedding;` for optimal ANN performance.
- [ ] **Upgrade LLM model** — Pull `qwen2.5:7b` or `mistral` for better answer quality once GPU budget allows.
- [ ] **Add conversation memory** — Implement multi-turn chat with sliding-window context from `query_history`.
- [ ] **Chunk overlap** — Add overlapping windows (e.g., 200 token overlap) during PDF chunking to avoid losing context at boundaries.

### Medium-Term
- [ ] **Dedicated cross-encoder** — Replace LLM re-ranking with a specialised model like `cross-encoder/ms-marco-MiniLM-L-6-v2` for faster and more accurate relevance scoring.
- [ ] **Evaluation framework** — Build a ground-truth Q&A dataset and measure Recall@k, MRR, BLEU, ROUGE, and latency benchmarks.
- [ ] **Metadata filtering** — Allow users to filter by document type, date range, or custom tags in queries.
- [ ] **Embedding cache** — Cache query embeddings in Redis to speed up repeated queries.
- [ ] **Chunk deduplication** — Detect and merge near-duplicate chunks across documents.
- [ ] **Streaming re-ranking** — Overlap re-ranking with LLM generation to reduce perceived latency.
- [ ] **Remove ChromaDB** — Since pgvector handles everything, remove ChromaDB from the stack to save resources.
- [ ] **Authentication & RBAC** — Add JWT-based auth with role-based access (admin, viewer, uploader).

### Long-Term
- [ ] **Domain-specific embedding model** — Fine-tune an embedding model on French real estate text for improved retrieval quality.
- [ ] **Multi-modal support** — Extract and index images, tables, and floor plans from PDFs.
- [ ] **Agent-based RAG** — Implement an agentic approach where the LLM decides when to search, when to ask clarifying questions, and when to use tools (calculator, map API).
- [ ] **Production deployment** — Kubernetes, TLS termination, monitoring (Prometheus/Grafana), rate limiting, structured logging pipeline.
- [ ] **Fine-tuned LLM** — LoRA/QLoRA fine-tuning of qwen2.5 or Mistral on the accumulated training examples for domain-specific generation quality.
- [ ] **Guardrails** — Add output validation to detect hallucinations, off-topic responses, and PII leakage.
- [ ] **A/B testing** — Compare pipeline configurations (with/without re-ranking, different models, different k values) with user feedback.

---

*Last updated: February 24, 2026*
