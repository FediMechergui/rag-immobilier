# 🏠 Immobilier RAG Pipeline

A production-ready **RAG (Retrieval-Augmented Generation)** pipeline specialized for the French real estate (immobilier) domain. This system allows you to upload PDF documents, ask questions in French, English, or Arabic, and get accurate answers with source citations.

## ✨ Features

- 📄 **PDF Document Processing**: Upload and process real estate documents
- 🔍 **Semantic Search**: Find relevant information using AI embeddings
- 💬 **Intelligent Q&A**: Get accurate answers with source citations
- 🌐 **Web Search**: Optional web search from trusted real estate sources
- 📚 **Training**: Improve responses with custom examples
- 🌍 **Multilingual**: Supports French, English, and Arabic
- 🐳 **Docker Compose**: Full containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   React Frontend │────│  FastAPI Backend │
│   (Port 3000)    │    │   (Port 8080)    │
└─────────────────┘     └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   PostgreSQL  │     │    Ollama     │     │   ChromaDB    │
│  + pgvector   │     │ (Local LLM)   │     │(Vector Store) │
│  (Port 5432)  │     │ (Port 11434)  │     │ (Port 8000)   │
└───────────────┘     └───────────────┘     └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- At least 8GB RAM (recommended 16GB for larger models)
- ~10GB disk space for Docker images and models

### 1. Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-immobilier

# Copy environment variables
cp .env.example .env

# Edit .env if needed (defaults work for local development)
```

### 2. Start Services

```bash
# Start all services
docker compose up -d

# Pull the Ollama model (first time only)
docker compose exec ollama ollama pull llama3.1

# Or for smaller model:
docker compose exec ollama ollama pull mistral
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

## 📁 Project Structure

```
rag-immobilier/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/
│   │   │       ├── documents.py   # Document upload/management
│   │   │       ├── query.py       # RAG query endpoints
│   │   │       └── training.py    # Training examples
│   │   ├── core/
│   │   │   ├── config.py          # Settings & configuration
│   │   │   ├── ollama_client.py   # Ollama LLM wrapper
│   │   │   └── prompts.py         # System prompts (FR/EN/AR)
│   │   ├── models/
│   │   │   ├── database.py        # SQLAlchemy models
│   │   │   └── schemas.py         # Pydantic schemas
│   │   ├── services/
│   │   │   ├── embeddings.py      # HuggingFace embeddings
│   │   │   ├── pdf_processor.py   # PDF extraction & chunking
│   │   │   ├── rag_pipeline.py    # RAG orchestration
│   │   │   └── web_search.py      # Web scraping
│   │   └── main.py                # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   └── init.sql                   # Database schema
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx           # Chat interface
│   │   │   ├── Documents.tsx      # Document management
│   │   │   ├── Settings.tsx       # Settings panel
│   │   │   └── Sidebar.tsx        # Navigation sidebar
│   │   ├── services/
│   │   │   └── api.ts             # API client
│   │   ├── store/
│   │   │   └── index.ts           # Zustand stores
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── i18n.ts                # Translations
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `ollama` | Ollama service hostname |
| `OLLAMA_PORT` | `11434` | Ollama service port |
| `OLLAMA_MODEL` | `llama3.1` | LLM model to use |
| `POSTGRES_HOST` | `postgres` | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `ragdb` | Database name |
| `POSTGRES_USER` | `raguser` | Database user |
| `POSTGRES_PASSWORD` | `ragpassword` | Database password |
| `CHROMA_HOST` | `chromadb` | ChromaDB hostname |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

### Ollama Models

You can use different models based on your hardware:

| Model | RAM Required | Speed | Quality |
|-------|-------------|-------|---------|
| `mistral` | ~8GB | Fast | Good |
| `llama3.1` | ~8GB | Medium | Better |
| `llama3.1:70b` | ~40GB | Slow | Best |

## 📖 API Reference

### Documents

- `POST /api/documents/upload` - Upload a PDF document
- `GET /api/documents/` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete a document

### Query

- `POST /api/query/` - Ask a question (non-streaming)
- `POST /api/query/stream` - Ask a question (streaming SSE)
- `POST /api/query/feedback` - Submit feedback

### Training

- `GET /api/training/` - List training examples
- `POST /api/training/` - Add training example
- `PUT /api/training/{id}` - Update training example
- `DELETE /api/training/{id}` - Delete training example

### Health

- `GET /health` - System health check

## 🌐 Web Search

The system can optionally search trusted real estate websites:

- seloger.com
- bienici.com
- notaires.fr
- service-public.fr
- legifrance.gouv.fr
- anil.org

Enable web search in the chat interface to include results from these sources.

## 🛠️ Development

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run with hot reload
uvicorn app.main:app --reload --port 8080
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## 📝 Adding Training Examples

Improve the system's responses by adding custom Q&A examples:

```bash
curl -X POST http://localhost:8080/api/training/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quels sont les frais de notaire pour un achat immobilier?",
    "answer": "Les frais de notaire représentent environ 7-8% du prix pour l ancien et 2-3% pour le neuf. Ils comprennent les droits de mutation, les émoluments du notaire et les frais administratifs.",
    "language": "fr"
  }'
```

## 🐛 Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is running
docker compose logs ollama

# Pull model manually
docker compose exec ollama ollama pull llama3.1
```

### PostgreSQL Connection Issues

```bash
# Check database logs
docker compose logs postgres

# Verify database is ready
docker compose exec postgres pg_isready
```

### Out of Memory

Reduce model size or increase Docker memory limit:

```yaml
# docker-compose.yml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 8G
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.
