"""
Rolling-window metrics tracker for the router.
"""
import time
from collections import deque
from typing import Dict, List, Deque, Tuple
import statistics

from src.models import MetricsSnapshot


class MetricsTracker:
    """Tracks request metrics using a rolling window."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: Deque[float] = deque(maxlen=window_size)
        self._model_usage: Dict[str, int] = {}
        self._model_latencies: Dict[str, List[float]] = {}
        self._total_requests: int = 0
        self._cached_requests: int = 0
        self._errors: int = 0

    def record(self, model: str, latency_ms: float, cached: bool = False, error: bool = False):
        self._total_requests += 1
        self._latencies.append(latency_ms)

        if cached:
            self._cached_requests += 1
        if error:
            self._errors += 1

        self._model_usage[model] = self._model_usage.get(model, 0) + 1
        self._model_latencies.setdefault(model, [])
        self._model_latencies[model].append(latency_ms)
        if len(self._model_latencies[model]) > self.window_size:
            self._model_latencies[model] = self._model_latencies[model][-self.window_size:]

    def snapshot(self) -> MetricsSnapshot:
        latencies = list(self._latencies) or [0.0]
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        avg = statistics.mean(sorted_lat)
        p95 = sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[-1]
        p99 = sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[-1]

        model_avg_latencies = {
            m: statistics.mean(lats) for m, lats in self._model_latencies.items() if lats
        }

        return MetricsSnapshot(
            total_requests=self._total_requests,
            cached_requests=self._cached_requests,
            avg_latency_ms=round(avg, 2),
            p95_latency_ms=round(p95, 2),
            p99_latency_ms=round(p99, 2),
            errors_count=self._errors,
            model_usage=dict(self._model_usage),
            model_latencies=model_avg_latencies,
        )

    def reset(self):
        self._latencies.clear()
        self._model_usage.clear()
        self._model_latencies.clear()
        self._total_requests = 0
        self._cached_requests = 0
        self._errors = 0
