"""
PDF Processing Service
Handles PDF ingestion, text extraction, and chunking.
"""
import os
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect, DetectorFactory
import tiktoken
import structlog

from app.core.config import get_settings

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = structlog.get_logger()
settings = get_settings()


@dataclass
class ExtractedPage:
    """Represents an extracted page from a PDF."""
    page_number: int
    text: str
    metadata: Dict[str, Any]


@dataclass
class DocumentChunk:
    """Represents a chunk of document text."""
    content: str
    page_number: Optional[int]
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any]


class PDFProcessor:
    """Process PDF documents for the RAG pipeline."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize text splitter with multilingual separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", "،", "。", ";", ":", " ", ""],
            length_function=self._count_tokens
        )
        
        # Token encoder for accurate counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
        
        logger.info(
            "PDF Processor initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback: estimate 4 chars per token
        return len(text) // 4
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (fr, en, ar, or 'unknown')
        """
        try:
            if len(text.strip()) < 50:
                return "unknown"
            # Use first 1000 chars for detection
            sample = text[:1000]
            lang = detect(sample)
            return lang if lang in ["fr", "en", "ar"] else "unknown"
        except Exception:
            return "unknown"
    
    def extract_with_pymupdf(self, file_path: str) -> List[ExtractedPage]:
        """
        Extract text using PyMuPDF (fast, good for standard PDFs).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of ExtractedPage objects
        """
        pages = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                # Get page metadata
                metadata = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                }
                
                pages.append(ExtractedPage(
                    page_number=page_num + 1,
                    text=text,
                    metadata=metadata
                ))
            doc.close()
        except Exception as e:
            logger.error("PyMuPDF extraction failed", error=str(e), file=file_path)
        
        return pages
    
    def extract_with_pdfplumber(self, file_path: str) -> List[ExtractedPage]:
        """
        Extract text using pdfplumber (better for complex layouts).
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of ExtractedPage objects
        """
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Get page metadata
                    metadata = {
                        "width": page.width,
                        "height": page.height
                    }
                    
                    # Try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        metadata["has_tables"] = True
                        metadata["table_count"] = len(tables)
                    
                    pages.append(ExtractedPage(
                        page_number=page_num + 1,
                        text=text,
                        metadata=metadata
                    ))
        except Exception as e:
            logger.error("pdfplumber extraction failed", error=str(e), file=file_path)
        
        return pages
    
    def extract_text(self, file_path: str) -> Tuple[List[ExtractedPage], int]:
        """
        Extract text from PDF using the best available method.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (list of ExtractedPage, total page count)
        """
        # Try PyMuPDF first (faster)
        pages = self.extract_with_pymupdf(file_path)
        
        # Calculate total text length
        total_text = sum(len(p.text) for p in pages)
        
        # If little text extracted, try pdfplumber
        if total_text < 100:
            pages_plumber = self.extract_with_pdfplumber(file_path)
            total_plumber = sum(len(p.text) for p in pages_plumber)
            
            if total_plumber > total_text:
                pages = pages_plumber
                logger.info("Using pdfplumber extraction (better results)")
        
        return pages, len(pages)
    
    def chunk_document(
        self,
        pages: List[ExtractedPage],
        filename: str
    ) -> List[DocumentChunk]:
        """
        Split document pages into chunks.
        
        Args:
            pages: List of extracted pages
            filename: Original filename for metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_index = 0
        
        for page in pages:
            if not page.text.strip():
                continue
            
            # Split page text into chunks
            page_chunks = self.text_splitter.split_text(page.text)
            
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                
                token_count = self._count_tokens(chunk_text)
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata={
                        "filename": filename,
                        "page": page.page_number,
                        **page.metadata
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def process_pdf(
        self,
        file_path: str,
        filename: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Process a PDF file: extract text, detect language, and create chunks.
        
        Args:
            file_path: Path to PDF file
            filename: Original filename
            
        Returns:
            Tuple of (list of chunks, document metadata)
        """
        logger.info("Processing PDF", filename=filename)
        
        # Extract text
        pages, page_count = self.extract_text(file_path)
        
        if not pages:
            logger.warning("No text extracted from PDF", filename=filename)
            return [], {"page_count": 0, "language": "unknown"}
        
        # Combine text for language detection
        full_text = " ".join(p.text for p in pages[:3])  # Use first 3 pages
        language = self.detect_language(full_text)
        
        # Create chunks
        chunks = self.chunk_document(pages, filename)
        
        # Document metadata
        doc_metadata = {
            "page_count": page_count,
            "language": language,
            "chunk_count": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks)
        }
        
        logger.info(
            "PDF processed successfully",
            filename=filename,
            pages=page_count,
            chunks=len(chunks),
            language=language
        )
        
        return chunks, doc_metadata
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a file is a valid PDF.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, "File not found"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = settings.max_upload_size_mb * 1024 * 1024
        if file_size > max_size:
            return False, f"File too large (max {settings.max_upload_size_mb}MB)"
        
        # Check if valid PDF
        try:
            doc = fitz.open(file_path)
            if doc.page_count == 0:
                return False, "PDF has no pages"
            doc.close()
            return True, None
        except Exception as e:
            return False, f"Invalid PDF: {str(e)}"


# Singleton instance
_pdf_processor: Optional[PDFProcessor] = None


def get_pdf_processor() -> PDFProcessor:
    """Get or create PDF processor singleton."""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor
