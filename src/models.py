"""
Pydantic data models for the Low-Latency Model Router.
"""
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class Message(BaseModel):
    """Chat message format."""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Message content")


class RouteRequest(BaseModel):
    """Request to route a completion/chat request to the optimal model."""
    model_config = ConfigDict(extra="allow")
    
    messages: List[Message] = Field(..., description="List of messages for chat completion")
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    
    # Routing preferences
    priority: Optional[str] = Field(default="balanced", description="Priority: 'speed', 'cost', 'quality', or 'balanced'")
    preferred_models: Optional[List[str]] = Field(default=None, description="List of preferred model IDs")
    max_latency_ms: Optional[int] = Field(default=None, description="Maximum acceptable latency in milliseconds")
    max_cost_per_1k: Optional[float] = Field(default=None, description="Maximum cost per 1K tokens")


class RouteResponse(BaseModel):
    """Response from the router with completion and routing metadata."""
    id: str = Field(..., description="Unique response ID")
    model: str = Field(..., description="Model ID that was used")
    choices: List[Dict[str, Any]] = Field(..., description="Completion choices")
    usage: Dict[str, Any] = Field(..., description="Token usage statistics")
    
    # Routing metadata
    routing_decision: Dict[str, Any] = Field(..., description="Details about the routing decision")
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    cached: bool = Field(default=False, description="Whether response was served from cache")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelMetadata(BaseModel):
    """Metadata about a model available through OpenRouter."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    provider: str = Field(..., description="Model provider (openai, anthropic, etc.)")
    
    # Performance characteristics
    avg_latency_ms: Optional[float] = Field(default=None, description="Average latency in ms")
    p95_latency_ms: Optional[float] = Field(default=None, description="95th percentile latency")
    
    # Cost information (per 1K tokens)
    prompt_cost_per_1k: Optional[float] = Field(default=None)
    completion_cost_per_1k: Optional[float] = Field(default=None)
    
    # Capabilities
    context_length: Optional[int] = Field(default=None)
    supports_streaming: bool = Field(default=True)
    supports_vision: bool = Field(default=False)
    
    # Quality metrics
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Last updated
    last_updated: Optional[datetime] = Field(default=None)


class RoutingWeights(BaseModel):
    """Weights for the routing algorithm."""
    latency: float = Field(default=0.4, ge=0.0, le=1.0)
    cost: float = Field(default=0.3, ge=0.0, le=1.0)
    quality: float = Field(default=0.3, ge=0.0, le=1.0)
    
    def normalize(self) -> "RoutingWeights":
        """Normalize weights to sum to 1.0."""
        total = self.latency + self.cost + self.quality
        if total == 0:
            return RoutingWeights(latency=0.4, cost=0.3, quality=0.3)
        return RoutingWeights(
            latency=self.latency / total,
            cost=self.cost / total,
            quality=self.quality / total
        )


class ModelScore(BaseModel):
    """Score for a model in routing decision."""
    model_id: str
    latency_score: float
    cost_score: float
    quality_score: float
    composite_score: float
    selected: bool = False


class RoutingDecision(BaseModel):
    """Detailed routing decision information."""
    selected_model: str
    weights_used: RoutingWeights
    candidate_scores: List[ModelScore]
    reason: str
    estimated_latency_ms: Optional[float] = None
    estimated_cost: Optional[float] = None


class MetricsSnapshot(BaseModel):
    """Snapshot of current router metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_requests: int
    cached_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors_count: int
    
    # Per-model stats
    model_usage: Dict[str, int]
    model_latencies: Dict[str, float]


class CacheEntry(BaseModel):
    """Cache entry structure."""
    key: str
    response: RouteResponse
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(default=3600)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    redis_connected: bool
    openrouter_accessible: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")
