"""
Embedding Service
Handles text embedding generation using HuggingFace models.
"""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model name (default from settings)
        """
        self.model_name = model_name or settings.hf_model
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension: int = 768  # Default for multilingual models
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info("Loading embedding model", model=self.model_name)
            self.model = SentenceTransformer(self.model_name)
            
            # Get actual embedding dimension
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dimension = len(test_embedding)
            
            logger.info(
                "Embedding model loaded",
                model=self.model_name,
                dimension=self.embedding_dimension
            )
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e))
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error("Batch embedding generation failed", error=str(e))
            raise
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of embeddings to search
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        embeddings_array = np.array(embeddings)
        
        # Compute all similarities at once
        similarities = np.dot(embeddings_array, query_vec)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
