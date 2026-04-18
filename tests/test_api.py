"""Integration tests for the FastAPI routes using mocked OpenRouter."""
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.api.main import app
from src.models import RouteResponse


def _mock_response(model="openai/gpt-4o-mini") -> RouteResponse:
    return RouteResponse(
        id="mock-id",
        model=model,
        choices=[{"message": {"role": "assistant", "content": "Mock answer"}}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        routing_decision={"selected_model": model, "reason": "mock"},
        latency_ms=42.0,
        cached=False,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "redis_connected" in data
    assert "openrouter_accessible" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all("id" in m for m in models)


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data


def test_cache_stats_endpoint(client):
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "connected" in data


def test_cache_clear_endpoint(client):
    response = client.delete("/cache")
    assert response.status_code == 200
    assert "message" in response.json()


@patch("src.api.routes.OpenRouterClient")
def test_route_endpoint_mocked(MockClient, client):
    mock_instance = AsyncMock()
    mock_instance.chat_completion = AsyncMock(return_value=_mock_response())
    mock_instance.health_check = AsyncMock(return_value=True)
    mock_instance.close = AsyncMock()
    MockClient.return_value = mock_instance

    payload = {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "priority": "balanced",
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "choices" in data
    assert "routing_decision" in data
    assert "latency_ms" in data


@patch("src.api.routes.OpenRouterClient")
def test_route_fallback_on_error(MockClient, client):
    from src.router.openrouter import OpenRouterError

    mock_instance = AsyncMock()
    mock_instance.chat_completion = AsyncMock(
        side_effect=[
            OpenRouterError("Primary failed", status_code=500, latency_ms=10.0),
            _mock_response("anthropic/claude-3-haiku"),
        ]
    )
    mock_instance.close = AsyncMock()
    MockClient.return_value = mock_instance

    payload = {
        "messages": [{"role": "user", "content": "Test fallback"}],
        "priority": "balanced",
    }
    response = client.post("/route", json=payload)
    assert response.status_code == 200
