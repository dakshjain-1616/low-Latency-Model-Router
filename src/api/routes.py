"""
FastAPI route definitions.
"""
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.models import (
    RouteRequest, RouteResponse, ModelMetadata, MetricsSnapshot, HealthStatus
)
from src.router.openrouter import OpenRouterClient, OpenRouterError

router = APIRouter()


def _get_app_state():
    from src.api.main import router_instance, cache_instance, metrics_instance
    return router_instance, cache_instance, metrics_instance


@router.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    model_router, cache, _ = _get_app_state()
    client = OpenRouterClient()
    or_ok = await client.health_check()
    await client.close()
    return HealthStatus(
        status="healthy",
        redis_connected=cache.is_connected(),
        openrouter_accessible=or_ok,
    )


@router.post("/route", response_model=RouteResponse, tags=["Routing"])
async def route_request(request: RouteRequest):
    """
    Route a chat completion request to the optimal model.
    Uses weighted scoring across latency, cost, and quality.
    Falls back automatically on model failure.
    """
    model_router, cache, metrics = _get_app_state()

    decision = model_router.route(request)
    selected_model = decision.selected_model

    messages_dict = [m.model_dump() for m in request.messages]
    params = {"max_tokens": request.max_tokens, "temperature": request.temperature}

    cached = cache.get(selected_model, messages_dict, params)
    if cached:
        metrics.record(selected_model, cached.latency_ms, cached=True)
        return cached

    client = OpenRouterClient()
    try:
        response = await client.chat_completion(selected_model, request, decision)
        cache.set(selected_model, messages_dict, response, params)
        metrics.record(selected_model, response.latency_ms)
        return response
    except OpenRouterError as e:
        metrics.record(selected_model, e.latency_ms or 0, error=True)
        fallback = model_router.get_fallback(selected_model, request)
        if not fallback:
            raise HTTPException(status_code=502, detail=str(e))
        try:
            response = await client.chat_completion(fallback, request)
            metrics.record(fallback, response.latency_ms)
            return response
        except OpenRouterError as e2:
            metrics.record(fallback, e2.latency_ms or 0, error=True)
            raise HTTPException(status_code=502, detail=f"All models failed. Last: {e2}")
    finally:
        await client.close()


@router.get("/models", response_model=List[ModelMetadata], tags=["Models"])
async def list_models():
    """List all models known to the router with their metadata."""
    model_router, _, _ = _get_app_state()
    return list(model_router.models.values())


@router.get("/metrics", response_model=MetricsSnapshot, tags=["Metrics"])
async def get_metrics():
    """Return current rolling-window metrics snapshot."""
    _, _, metrics = _get_app_state()
    return metrics.snapshot()


@router.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Invalidate all cached responses."""
    _, cache, _ = _get_app_state()
    count = cache.invalidate()
    return {"message": f"Cleared {count} cache entries."}


@router.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Return cache statistics."""
    _, cache, _ = _get_app_state()
    return cache.get_stats()
