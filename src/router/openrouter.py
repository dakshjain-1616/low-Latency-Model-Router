"""
OpenRouter API client for making model inference requests.
"""
import os
import time
import uuid
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

from src.models import RouteRequest, RouteResponse, RoutingDecision


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 30.0
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://model-router.local",
            "X-Title": "Low-Latency Model Router"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Fetch list of available models from OpenRouter."""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    async def chat_completion(
        self,
        model: str,
        request: RouteRequest,
        routing_decision: Optional[RoutingDecision] = None
    ) -> RouteResponse:
        """
        Send a chat completion request to OpenRouter.
        
        Args:
            model: Model ID to use
            request: RouteRequest with messages and parameters
            routing_decision: Optional routing decision metadata
            
        Returns:
            RouteResponse with completion and metadata
        """
        start_time = time.time()
        
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 1.0,
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Build routing decision dict
            decision_dict = {}
            if routing_decision:
                decision_dict = {
                    "selected_model": routing_decision.selected_model,
                    "weights_used": routing_decision.weights_used.model_dump(),
                    "reason": routing_decision.reason
                }
            else:
                decision_dict = {"selected_model": model, "reason": "direct request"}
            
            return RouteResponse(
                id=data.get("id", str(uuid.uuid4())),
                model=model,
                choices=data.get("choices", []),
                usage=data.get("usage", {}),
                routing_decision=decision_dict,
                latency_ms=latency_ms,
                cached=False,
                timestamp=datetime.utcnow()
            )
            
        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            raise OpenRouterError(
                f"HTTP error {e.response.status_code}: {e.response.text}",
                status_code=e.response.status_code,
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise OpenRouterError(str(e), latency_ms=latency_ms)
    
    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible."""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False


class OpenRouterError(Exception):
    """Custom exception for OpenRouter errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.latency_ms = latency_ms
