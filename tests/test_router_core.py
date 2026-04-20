"""Tests for the core routing engine."""
import pytest
from src.router.core import ModelRouter, DEFAULT_MODELS
from src.models import RouteRequest, Message, ModelMetadata


def make_request(priority="balanced", **kwargs) -> RouteRequest:
    return RouteRequest(
        messages=[Message(role="user", content="Hello")],
        priority=priority,
        **kwargs
    )


def test_route_returns_decision():
    router = ModelRouter()
    decision = router.route(make_request())
    assert decision.selected_model in router.models
    assert len(decision.candidate_scores) > 0


def test_speed_priority_selects_low_latency():
    router = ModelRouter()
    decision = router.route(make_request(priority="speed"))
    selected = router.models[decision.selected_model]
    # The lowest latency model should score highest
    assert selected.avg_latency_ms == min(m.avg_latency_ms for m in DEFAULT_MODELS)


def test_quality_priority_selects_high_quality():
    router = ModelRouter()
    decision = router.route(make_request(priority="quality"))
    selected = router.models[decision.selected_model]
    # Quality weight is 0.7 — selected model must have a respectable quality score
    assert selected.quality_score >= 0.75
    # And must score better than a speed-priority choice on quality
    speed_decision = router.route(make_request(priority="speed"))
    speed_selected = router.models[speed_decision.selected_model]
    assert selected.quality_score >= speed_selected.quality_score


def test_cost_priority():
    router = ModelRouter()
    decision = router.route(make_request(priority="cost"))
    selected = router.models[decision.selected_model]
    cost = (selected.prompt_cost_per_1k or 0) + (selected.completion_cost_per_1k or 0)
    assert cost < 0.01  # Should select a cheap model


def test_max_latency_filter():
    router = ModelRouter()
    decision = router.route(make_request(max_latency_ms=700))
    selected = router.models[decision.selected_model]
    assert (selected.avg_latency_ms or 9999) <= 700


def test_preferred_models():
    router = ModelRouter()
    preferred = ["anthropic/claude-opus-4.7"]
    decision = router.route(make_request(preferred_models=preferred))
    assert decision.selected_model == "anthropic/claude-opus-4.7"


def test_composite_scores_sum_correctly():
    router = ModelRouter()
    decision = router.route(make_request())
    for score in decision.candidate_scores:
        assert 0.0 <= score.composite_score <= 1.0


def test_fallback_on_missing_preferred():
    router = ModelRouter()
    decision = router.route(make_request(preferred_models=["nonexistent/model"]))
    # Falls back to all models
    assert decision.selected_model in router.models


def test_get_fallback():
    router = ModelRouter()
    fallback = router.get_fallback("openai/gpt-5.4-mini", make_request())
    assert fallback != "openai/gpt-5.4-mini"
    assert fallback in router.models


def test_weights_normalize():
    from src.models import RoutingWeights
    # Use valid values (each <=1) that sum to more than 1 before normalization
    w = RoutingWeights(latency=0.6, cost=0.6, quality=0.6)
    n = w.normalize()
    assert abs(n.latency + n.cost + n.quality - 1.0) < 1e-9
