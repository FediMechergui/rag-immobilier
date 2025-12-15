"""
Ollama client wrapper for LangChain integration.
"""
import httpx
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import structlog

from app.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Stop sequences to prevent repetition and hallucination
STOP_SEQUENCES = [
    "</s>",
    "[/INST]",
    "\n\nQuestion:",
    "\n\nQUESTION:",
    "\n\nHuman:",
    "\n\nUser:",
    "--- Extrait",  # Stop if model tries to repeat context
    "\n\nCONTEXTE",
    "\n\nSi vous avez",  # Common repetition pattern
    "En suivant ces",  # Another common ending
]


class OllamaClient:
    """Wrapper for Ollama LLM integration."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (default from settings)
            base_url: Ollama server URL (default from settings)
            temperature: Generation temperature
        """
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature
        
        # Initialize LangChain Ollama wrapper with stop tokens and limits
        self.llm = Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=800,  # Reduced to prevent long repetitive outputs
            repeat_penalty=1.2,  # Penalize repetition
            stop=STOP_SEQUENCES
        )
        
        # Chat model for conversational interactions
        self.chat_model = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=800,
            repeat_penalty=1.2,
            stop=STOP_SEQUENCES
        )
        
        logger.info("Ollama client initialized", model=self.model, base_url=self.base_url)
    
    def _clean_response(self, text: str) -> str:
        """Clean up model response to remove artifacts and repetition."""
        if not text:
            return text
        
        # Remove any remaining source markers the model might output
        text = re.sub(r'\[Source:.*?\]', '', text)
        text = re.sub(r'\[Web:.*?\]', '', text)
        text = re.sub(r'\[Page:.*?\]', '', text)
        text = re.sub(r'\[Document:.*?\]', '', text)
        
        # Detect and remove repeated paragraphs
        paragraphs = text.split('\n\n')
        seen = set()
        unique_paragraphs = []
        for p in paragraphs:
            p_normalized = ' '.join(p.split()).lower()
            if p_normalized and p_normalized not in seen:
                seen.add(p_normalized)
                unique_paragraphs.append(p)
        
        text = '\n\n'.join(unique_paragraphs)
        
        # Remove trailing incomplete sentences
        if text and not text.rstrip().endswith(('.', '!', '?', ':', ';')):
            last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_period > len(text) * 0.7:  # Only trim if we keep most of the content
                text = text[:last_period + 1]
        
        return text.strip()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check if Ollama server is healthy and model is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check server health
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                
                model_available = any(self.model in m for m in models)
                
                return {
                    "status": "healthy",
                    "server_url": self.base_url,
                    "model": self.model,
                    "model_available": model_available,
                    "available_models": models
                }
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "server_url": self.base_url,
                "model": self.model
            }
    
    async def pull_model(self, model_name: Optional[str] = None) -> bool:
        """Pull a model from Ollama registry."""
        model = model_name or self.model
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model}
                )
                response.raise_for_status()
                logger.info("Model pulled successfully", model=model)
                return True
        except Exception as e:
            logger.error("Failed to pull model", model=model, error=str(e))
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            return self.llm.invoke(prompt, **kwargs)
        except Exception as e:
            logger.error("Generation failed", error=str(e))
            raise
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """
        Async generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text (cleaned)
        """
        try:
            result = await self.llm.ainvoke(prompt, **kwargs)
            return self._clean_response(result)
        except Exception as e:
            logger.error("Async generation failed", error=str(e))
            raise
    
    async def stream_generate(
        self, 
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        try:
            async for chunk in self.llm.astream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.error("Stream generation failed", error=str(e))
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            
        Returns:
            Assistant response
        """
        langchain_messages = []
        
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
        
        try:
            response = self.chat_model.invoke(langchain_messages)
            return response.content
        except Exception as e:
            logger.error("Chat failed", error=str(e))
            raise
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Async chat completion."""
        langchain_messages = []
        
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
        
        try:
            response = await self.chat_model.ainvoke(langchain_messages)
            return self._clean_response(response.content)
        except Exception as e:
            logger.error("Async chat failed", error=str(e))
            raise


# Singleton instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
