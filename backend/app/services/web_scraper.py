"""
Web Scraping Service
Real web scrapers for French real estate websites.
"""
import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin, quote_plus
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


class RealEstateScraper:
    """Base class for real estate website scrapers."""
    
    def __init__(self):
        self.timeout = httpx.Timeout(15.0, connect=10.0)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def fetch(self, url: str) -> Optional[str]:
        """Fetch page content with retries."""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    http2=True
                ) as client:
                    response = await client.get(url, headers=self.headers)
                    if response.status_code == 200:
                        return response.text
                    elif response.status_code == 429:  # Rate limited
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.warning(f"HTTP {response.status_code} for {url}")
                        return None
            except Exception as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class ServicePublicScraper(RealEstateScraper):
    """Scraper for service-public.fr (official French government site)."""
    
    BASE_URL = "https://www.service-public.fr"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search service-public.fr for relevant information."""
        results = []
        search_url = f"{self.BASE_URL}/recherche?keyword={quote_plus(query)}&rubrique=particuliers"
        
        html = await self.fetch(search_url)
        if not html:
            return results
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Find search results
        result_items = soup.select(".search-results-list .search-result-item, .sp-liste-resultat li")[:max_results]
        
        for item in result_items:
            try:
                link = item.select_one("a")
                if not link:
                    continue
                
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(self.BASE_URL, href)
                
                title = link.get_text(strip=True)
                
                # Get snippet if available
                snippet = ""
                snippet_elem = item.select_one(".search-result-snippet, p")
                if snippet_elem:
                    snippet = snippet_elem.get_text(strip=True)
                
                results.append({
                    "url": href,
                    "title": title,
                    "snippet": snippet,
                    "domain": "service-public.fr",
                    "source": "government"
                })
            except Exception as e:
                logger.warning(f"Error parsing result: {e}")
        
        return results
    
    async def get_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Get full article content from service-public.fr."""
        html = await self.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for elem in soup.select("nav, footer, .breadcrumb, .share-buttons, script, style"):
            elem.decompose()
        
        title = ""
        title_elem = soup.select_one("h1, .sp-titre")
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        content = ""
        content_elem = soup.select_one("article, .sp-article-content, main .content")
        if content_elem:
            content = self.clean_text(content_elem.get_text(separator="\n"))
        
        return {
            "url": url,
            "title": title,
            "content": content[:5000],  # Limit content
            "domain": "service-public.fr",
            "retrieved_date": datetime.utcnow().isoformat()
        }


class LegiFranceScraper(RealEstateScraper):
    """Scraper for legifrance.gouv.fr (French legal texts)."""
    
    BASE_URL = "https://www.legifrance.gouv.fr"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search legifrance for relevant legal texts."""
        results = []
        search_url = f"{self.BASE_URL}/search/all?text={quote_plus(query)}&tab_selection=code"
        
        html = await self.fetch(search_url)
        if not html:
            return results
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Find search results
        result_items = soup.select(".result-item, .search-result")[:max_results]
        
        for item in result_items:
            try:
                link = item.select_one("a")
                if not link:
                    continue
                
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(self.BASE_URL, href)
                
                title = link.get_text(strip=True)
                
                results.append({
                    "url": href,
                    "title": title,
                    "domain": "legifrance.gouv.fr",
                    "source": "legal"
                })
            except Exception as e:
                logger.warning(f"Error parsing legifrance result: {e}")
        
        return results


class ANILScraper(RealEstateScraper):
    """Scraper for anil.org (French housing information agency)."""
    
    BASE_URL = "https://www.anil.org"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ANIL for housing-related information."""
        results = []
        search_url = f"{self.BASE_URL}/recherche?keywords={quote_plus(query)}"
        
        html = await self.fetch(search_url)
        if not html:
            return results
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Find search results
        result_items = soup.select(".views-row, .search-result, article")[:max_results]
        
        for item in result_items:
            try:
                link = item.select_one("a[href]")
                if not link:
                    continue
                
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(self.BASE_URL, href)
                
                title = link.get_text(strip=True)
                
                # Get summary
                summary = ""
                summary_elem = item.select_one(".summary, .teaser, p")
                if summary_elem:
                    summary = summary_elem.get_text(strip=True)
                
                results.append({
                    "url": href,
                    "title": title,
                    "snippet": summary,
                    "domain": "anil.org",
                    "source": "housing_agency"
                })
            except Exception as e:
                logger.warning(f"Error parsing ANIL result: {e}")
        
        return results
    
    async def get_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Get full article content from ANIL."""
        html = await self.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for elem in soup.select("nav, footer, .breadcrumb, script, style, .sidebar"):
            elem.decompose()
        
        title = ""
        title_elem = soup.select_one("h1, .page-title")
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        content = ""
        content_elem = soup.select_one("article, .content, .field--body")
        if content_elem:
            content = self.clean_text(content_elem.get_text(separator="\n"))
        
        return {
            "url": url,
            "title": title,
            "content": content[:5000],
            "domain": "anil.org",
            "retrieved_date": datetime.utcnow().isoformat()
        }


class NotairesScraper(RealEstateScraper):
    """Scraper for notaires.fr (French notaries official site)."""
    
    BASE_URL = "https://www.notaires.fr"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search notaires.fr for notary-related information."""
        results = []
        search_url = f"{self.BASE_URL}/rechercher?q={quote_plus(query)}"
        
        html = await self.fetch(search_url)
        if not html:
            return results
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Find search results
        result_items = soup.select(".search-result, .result-item, article")[:max_results]
        
        for item in result_items:
            try:
                link = item.select_one("a[href]")
                if not link:
                    continue
                
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = urljoin(self.BASE_URL, href)
                
                title = link.get_text(strip=True)
                
                results.append({
                    "url": href,
                    "title": title,
                    "domain": "notaires.fr",
                    "source": "notary"
                })
            except Exception as e:
                logger.warning(f"Error parsing notaires.fr result: {e}")
        
        return results


class WebScrapingService:
    """
    Main web scraping service that orchestrates multiple scrapers.
    """
    
    def __init__(self):
        self.scrapers = {
            "service-public.fr": ServicePublicScraper(),
            "legifrance.gouv.fr": LegiFranceScraper(),
            "anil.org": ANILScraper(),
            "notaires.fr": NotairesScraper(),
        }
        self.cache_ttl = settings.web_search_cache_ttl
        self.max_results = settings.web_search_max_results
        
        logger.info(f"Web Scraping Service initialized with {len(self.scrapers)} scrapers")
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
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
                logger.info("Web scraping cache hit", query_hash=query_hash[:16])
                return cached.results
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
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
            await db.execute(
                delete(WebSearchCache).where(WebSearchCache.query_hash == query_hash)
            )
            
            cache_entry = WebSearchCache(
                query_hash=query_hash,
                query=query,
                results=results,
                expires_at=datetime.utcnow() + timedelta(seconds=self.cache_ttl)
            )
            db.add(cache_entry)
            await db.commit()
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    async def search(
        self,
        db: AsyncSession,
        query: str,
        sources: List[str] = None
    ) -> List[WebSource]:
        """
        Search multiple sources for relevant information.
        
        Args:
            db: Database session
            query: Search query
            sources: Optional list of specific sources to search
            
        Returns:
            List of WebSource objects
        """
        query_hash = self._hash_query(query)
        
        # Check cache first
        cached = await self._check_cache(db, query_hash)
        if cached:
            return [
                WebSource(
                    type="web",
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    domain=r.get("domain", ""),
                    retrieved_date=datetime.fromisoformat(r.get("retrieved_date", datetime.utcnow().isoformat())),
                    relevance_score=r.get("relevance_score", 0.5)
                )
                for r in cached
            ]
        
        # Determine which scrapers to use
        scrapers_to_use = self.scrapers
        if sources:
            scrapers_to_use = {k: v for k, v in self.scrapers.items() if k in sources}
        
        # Search all scrapers in parallel
        all_results = []
        tasks = []
        
        for domain, scraper in scrapers_to_use.items():
            tasks.append(self._search_with_scraper(domain, scraper, query))
        
        if tasks:
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
                elif isinstance(results, Exception):
                    logger.warning(f"Scraper error: {results}")
        
        # Limit total results
        all_results = all_results[:self.max_results]
        
        # Save to cache
        if all_results:
            await self._save_cache(db, query, query_hash, all_results)
        
        # Convert to WebSource objects
        web_sources = []
        for r in all_results:
            try:
                web_sources.append(WebSource(
                    type="web",
                    title=r.get("title", "Unknown"),
                    url=r.get("url", ""),
                    domain=r.get("domain", "unknown"),
                    retrieved_date=datetime.utcnow(),
                    relevance_score=0.5  # Default score
                ))
            except Exception as e:
                logger.warning(f"Error creating WebSource: {e}")
        
        logger.info(f"Web search completed", query=query[:50], results=len(web_sources))
        return web_sources
    
    async def _search_with_scraper(
        self,
        domain: str,
        scraper: RealEstateScraper,
        query: str
    ) -> List[Dict[str, Any]]:
        """Search with a specific scraper."""
        try:
            results = await scraper.search(query, max_results=3)
            return results
        except Exception as e:
            logger.warning(f"Scraper {domain} failed: {e}")
            return []
    
    async def get_article_content(
        self,
        url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get full article content from a URL.
        
        Args:
            url: Article URL
            
        Returns:
            Article content dict or None
        """
        # Determine which scraper to use based on domain
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        scraper = self.scrapers.get(domain)
        if scraper and hasattr(scraper, "get_article"):
            return await scraper.get_article(url)
        
        # Fallback: use base scraper
        base_scraper = RealEstateScraper()
        html = await base_scraper.fetch(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for elem in soup.select("nav, footer, script, style, aside"):
            elem.decompose()
        
        title = ""
        title_elem = soup.select_one("h1, title")
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        content = ""
        content_elem = soup.select_one("article, main, .content, body")
        if content_elem:
            content = base_scraper.clean_text(content_elem.get_text(separator="\n"))
        
        return {
            "url": url,
            "title": title,
            "content": content[:5000],
            "domain": domain,
            "retrieved_date": datetime.utcnow().isoformat()
        }


# Singleton
_web_scraping_service: Optional[WebScrapingService] = None


def get_web_scraping_service() -> WebScrapingService:
    """Get or create web scraping service singleton."""
    global _web_scraping_service
    if _web_scraping_service is None:
        _web_scraping_service = WebScrapingService()
    return _web_scraping_service
