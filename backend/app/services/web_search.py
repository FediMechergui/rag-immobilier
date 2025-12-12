"""
Web Search Service
Handles web scraping from whitelisted real estate sources.
"""
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import httpx
from bs4 import BeautifulSoup
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.config import get_settings, WHITELISTED_DOMAINS
from app.models.database import WebSearchCache
from app.models.schemas import WebSource

logger = structlog.get_logger()
settings = get_settings()


class WebSearchService:
    """Service for searching and scraping whitelisted web sources."""
    
    def __init__(self):
        """Initialize web search service."""
        self.whitelisted_domains = WHITELISTED_DOMAINS
        self.max_results = settings.web_search_max_results
        self.cache_ttl = settings.web_search_cache_ttl
        
        # HTTP client settings
        self.timeout = httpx.Timeout(10.0, connect=5.0)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"
        }
        
        logger.info(
            "Web Search Service initialized",
            whitelisted_domains=len(self.whitelisted_domains)
        )
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
    def _is_whitelisted(self, url: str) -> bool:
        """Check if URL is from a whitelisted domain."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return any(domain.endswith(wd) for wd in self.whitelisted_domains)
        except Exception:
            return False
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.warning("Failed to fetch page", url=url, error=str(e))
            return None
    
    def _extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract relevant content from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Try to find main content
        main_content = ""
        
        # Try common content selectors
        content_selectors = [
            "article",
            "main",
            ".content",
            ".article-content",
            ".post-content",
            "#content",
            ".entry-content"
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element.get_text(separator=" ", strip=True)
                break
        
        # Fallback to body
        if not main_content:
            body = soup.find("body")
            if body:
                main_content = body.get_text(separator=" ", strip=True)
        
        # Clean and truncate content
        main_content = " ".join(main_content.split())  # Normalize whitespace
        main_content = main_content[:3000]  # Limit content length
        
        return {
            "title": title,
            "content": main_content,
            "url": url,
            "domain": self._extract_domain(url)
        }
    
    async def _check_cache(
        self,
        db: AsyncSession,
        query_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Check if results are cached."""
        try:
            result = await db.execute(
                select(WebSearchCache).where(
                    WebSearchCache.query_hash == query_hash,
                    WebSearchCache.expires_at > datetime.utcnow()
                )
            )
            cached = result.scalar_one_or_none()
            if cached:
                logger.info("Cache hit for web search", query_hash=query_hash[:16])
                return cached.results
        except Exception as e:
            logger.warning("Cache check failed", error=str(e))
        return None
    
    async def _save_cache(
        self,
        db: AsyncSession,
        query: str,
        query_hash: str,
        results: List[Dict[str, Any]]
    ):
        """Save results to cache."""
        try:
            # Delete old cache entry if exists
            await db.execute(
                delete(WebSearchCache).where(WebSearchCache.query_hash == query_hash)
            )
            
            # Create new cache entry
            cache_entry = WebSearchCache(
                query_hash=query_hash,
                query=query,
                results=results,
                expires_at=datetime.utcnow() + timedelta(seconds=self.cache_ttl)
            )
            db.add(cache_entry)
            await db.commit()
        except Exception as e:
            logger.warning("Cache save failed", error=str(e))
    
    def _build_search_urls(self, query: str) -> List[str]:
        """
        Build search URLs for whitelisted domains.
        
        Note: In a production system, you would use a proper search API
        or implement domain-specific search endpoints.
        """
        # For now, we'll construct potential URLs based on common patterns
        # In production, you'd want to use proper search APIs for each domain
        
        urls = []
        query_encoded = query.replace(" ", "+")
        
        # Domain-specific search patterns
        search_patterns = {
            "seloger.com": f"https://www.seloger.com/recherche.htm?q={query_encoded}",
            "bienici.com": f"https://www.bienici.com/recherche/{query_encoded}",
            "service-public.fr": f"https://www.service-public.fr/recherche?q={query_encoded}",
            "anil.org": f"https://www.anil.org/recherche/?q={query_encoded}",
        }
        
        for domain, url in search_patterns.items():
            urls.append(url)
        
        return urls[:self.max_results]
    
    async def search(
        self,
        db: AsyncSession,
        query: str,
        max_results: Optional[int] = None
    ) -> List[WebSource]:
        """
        Search whitelisted sources for relevant content.
        
        Args:
            db: Database session
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of WebSource objects
        """
        max_results = max_results or self.max_results
        query_hash = self._hash_query(query)
        
        logger.info("Performing web search", query=query[:50], max_results=max_results)
        
        # Check cache
        cached = await self._check_cache(db, query_hash)
        if cached:
            return [
                WebSource(
                    type="web",
                    title=r["title"],
                    url=r["url"],
                    domain=r["domain"],
                    retrieved_date=datetime.fromisoformat(r["retrieved_date"]),
                    relevance_score=r.get("relevance_score")
                )
                for r in cached[:max_results]
            ]
        
        # Build search URLs
        search_urls = self._build_search_urls(query)
        
        # Fetch and extract content concurrently
        results = []
        tasks = [self._fetch_page(url) for url in search_urls]
        pages = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url, page in zip(search_urls, pages):
            if isinstance(page, Exception) or not page:
                continue
            
            try:
                content = self._extract_content(page, url)
                if content["content"]:
                    results.append({
                        **content,
                        "retrieved_date": datetime.utcnow().isoformat()
                    })
            except Exception as e:
                logger.warning("Content extraction failed", url=url, error=str(e))
        
        # Save to cache
        if results:
            await self._save_cache(db, query, query_hash, results)
        
        # Convert to WebSource objects
        web_sources = [
            WebSource(
                type="web",
                title=r["title"],
                url=r["url"],
                domain=r["domain"],
                retrieved_date=datetime.fromisoformat(r["retrieved_date"]),
                relevance_score=None
            )
            for r in results[:max_results]
        ]
        
        logger.info("Web search completed", results_count=len(web_sources))
        
        return web_sources


# Singleton instance
_web_search_service: Optional[WebSearchService] = None


def get_web_search_service() -> WebSearchService:
    """Get or create web search service singleton."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
