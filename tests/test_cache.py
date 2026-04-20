"""Tests for the cache layer."""
import time
import pytest
from datetime import datetime, timezone
from src.router.cache import MockCache
from src.models import RouteResponse, RoutingWeights


def _make_response(model="openai/gpt-5.4-mini") -> RouteResponse:
    return RouteResponse(
        id="test-id",
        model=model,
        choices=[{"message": {"role": "assistant", "content": "Paris"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        routing_decision={"selected_model": model, "reason": "test"},
        latency_ms=100.0,
        cached=False,
        timestamp=datetime.now(timezone.utc),
    )


def test_mock_cache_set_get():
    cache = MockCache()
    messages = [{"role": "user", "content": "Hello"}]
    response = _make_response()
    cache.set("openai/gpt-5.4-mini", messages, response)
    result = cache.get("openai/gpt-5.4-mini", messages)
    assert result is not None
    assert result.cached is True
    assert result.model == "openai/gpt-5.4-mini"


def test_cache_miss():
    cache = MockCache()
    result = cache.get("openai/gpt-5.4-mini", [{"role": "user", "content": "xyz"}])
    assert result is None


def test_cache_different_params():
    cache = MockCache()
    messages = [{"role": "user", "content": "Hello"}]
    response = _make_response()
    cache.set("openai/gpt-5.4-mini", messages, response, params={"max_tokens": 100})
    result = cache.get("openai/gpt-5.4-mini", messages, params={"max_tokens": 200})
    assert result is None


def test_cache_invalidate():
    cache = MockCache()
    messages = [{"role": "user", "content": "Hello"}]
    cache.set("openai/gpt-5.4-mini", messages, _make_response())
    count = cache.invalidate()
    assert count == 1
    result = cache.get("openai/gpt-5.4-mini", messages)
    assert result is None


def test_cache_expiry():
    cache = MockCache(default_ttl=1)
    messages = [{"role": "user", "content": "Hello"}]
    cache.set("openai/gpt-5.4-mini", messages, _make_response(), ttl=1)
    time.sleep(1.1)
    result = cache.get("openai/gpt-5.4-mini", messages)
    assert result is None


def test_cache_stats():
    cache = MockCache()
    stats = cache.get_stats()
    assert stats["connected"] is True
    assert "cached_entries" in stats
