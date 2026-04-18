"""Tests for the metrics tracker."""
import pytest
from src.router.metrics import MetricsTracker


def test_initial_snapshot():
    tracker = MetricsTracker()
    snap = tracker.snapshot()
    assert snap.total_requests == 0
    assert snap.cached_requests == 0
    assert snap.errors_count == 0


def test_record_and_snapshot():
    tracker = MetricsTracker()
    tracker.record("openai/gpt-4o-mini", 100.0)
    tracker.record("openai/gpt-4o-mini", 200.0)
    tracker.record("anthropic/claude-3-haiku", 150.0, cached=True)

    snap = tracker.snapshot()
    assert snap.total_requests == 3
    assert snap.cached_requests == 1
    assert snap.errors_count == 0
    assert snap.avg_latency_ms == pytest.approx(150.0, abs=1)
    assert "openai/gpt-4o-mini" in snap.model_usage
    assert snap.model_usage["openai/gpt-4o-mini"] == 2


def test_error_recording():
    tracker = MetricsTracker()
    tracker.record("openai/gpt-4o", 50.0, error=True)
    snap = tracker.snapshot()
    assert snap.errors_count == 1


def test_percentiles():
    tracker = MetricsTracker()
    for i in range(1, 101):
        tracker.record("model", float(i))
    snap = tracker.snapshot()
    assert snap.avg_latency_ms == pytest.approx(50.5, abs=1)
    assert snap.p95_latency_ms >= 95.0


def test_reset():
    tracker = MetricsTracker()
    tracker.record("model", 100.0)
    tracker.reset()
    snap = tracker.snapshot()
    assert snap.total_requests == 0


def test_window_size():
    tracker = MetricsTracker(window_size=5)
    for i in range(10):
        tracker.record("model", float(i * 10))
    snap = tracker.snapshot()
    # Window of 5 keeps last 5 entries (50, 60, 70, 80, 90)
    assert snap.avg_latency_ms == pytest.approx(70.0, abs=1)
