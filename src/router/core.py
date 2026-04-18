"""
Core routing engine — selects optimal model using weighted scoring.
Score = w_latency*(1-norm_lat) + w_cost*(1-norm_cost) + w_quality*quality
"""
from typing import List, Optional, Dict, Any

from src.models import (
    ModelMetadata, RouteRequest, RoutingDecision, RoutingWeights, ModelScore
)


# Default model catalogue — verified against OpenRouter /api/v1/models on 2026-04-18
DEFAULT_MODELS: List[ModelMetadata] = [
    ModelMetadata(
        id="google/gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash Lite",
        provider="google",
        avg_latency_ms=400,
        p95_latency_ms=800,
        prompt_cost_per_1k=0.00025,
        completion_cost_per_1k=0.0015,
        context_length=1048576,
        quality_score=0.78,
    ),
    ModelMetadata(
        id="openai/gpt-5.4-mini",
        name="GPT-5.4 Mini",
        provider="openai",
        avg_latency_ms=700,
        p95_latency_ms=1400,
        prompt_cost_per_1k=0.00075,
        completion_cost_per_1k=0.0045,
        context_length=400000,
        quality_score=0.85,
    ),
    ModelMetadata(
        id="anthropic/claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        avg_latency_ms=900,
        p95_latency_ms=1800,
        prompt_cost_per_1k=0.003,
        completion_cost_per_1k=0.015,
        context_length=1000000,
        quality_score=0.92,
    ),
    ModelMetadata(
        id="google/gemini-3.1-pro-preview",
        name="Gemini 3.1 Pro",
        provider="google",
        avg_latency_ms=1100,
        p95_latency_ms=2200,
        prompt_cost_per_1k=0.002,
        completion_cost_per_1k=0.012,
        context_length=1048576,
        quality_score=0.93,
    ),
    ModelMetadata(
        id="openai/gpt-5.4",
        name="GPT-5.4",
        provider="openai",
        avg_latency_ms=1200,
        p95_latency_ms=2400,
        prompt_cost_per_1k=0.0025,
        completion_cost_per_1k=0.015,
        context_length=1050000,
        quality_score=0.96,
    ),
    ModelMetadata(
        id="anthropic/claude-opus-4.7",
        name="Claude Opus 4.7",
        provider="anthropic",
        avg_latency_ms=1500,
        p95_latency_ms=3000,
        prompt_cost_per_1k=0.005,
        completion_cost_per_1k=0.025,
        context_length=1000000,
        quality_score=0.98,
    ),
]

PRIORITY_WEIGHTS = {
    "speed":    RoutingWeights(latency=0.7, cost=0.2, quality=0.1),
    "cost":     RoutingWeights(latency=0.2, cost=0.7, quality=0.1),
    "quality":  RoutingWeights(latency=0.1, cost=0.2, quality=0.7),
    "balanced": RoutingWeights(latency=0.4, cost=0.3, quality=0.3),
}


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


class ModelRouter:
    """Selects the best model from a catalogue given routing preferences."""

    def __init__(
        self,
        models: Optional[List[ModelMetadata]] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.models: Dict[str, ModelMetadata] = {
            m.id: m for m in (models or DEFAULT_MODELS)
        }
        self.fallback_models = fallback_models or [
            "openai/gpt-5.4-mini",
            "google/gemini-3.1-flash-lite-preview",
            "anthropic/claude-sonnet-4.6",
        ]

    def update_model_metadata(self, metadata: List[ModelMetadata]):
        for m in metadata:
            self.models[m.id] = m

    def _get_weights(self, request: RouteRequest) -> RoutingWeights:
        weights = PRIORITY_WEIGHTS.get(request.priority or "balanced", PRIORITY_WEIGHTS["balanced"])
        return weights.normalize()

    def _get_candidates(self, request: RouteRequest) -> List[ModelMetadata]:
        candidates = list(self.models.values())

        if request.preferred_models:
            preferred = [m for m in candidates if m.id in request.preferred_models]
            if preferred:
                candidates = preferred

        if request.max_latency_ms is not None:
            candidates = [m for m in candidates if (m.avg_latency_ms or 9999) <= request.max_latency_ms]

        if request.max_cost_per_1k is not None:
            candidates = [
                m for m in candidates
                if (m.prompt_cost_per_1k or 0) + (m.completion_cost_per_1k or 0) <= request.max_cost_per_1k * 2
            ]

        return candidates or list(self.models.values())

    def _score_models(
        self, candidates: List[ModelMetadata], weights: RoutingWeights
    ) -> List[ModelScore]:
        latencies = [m.avg_latency_ms or 1000 for m in candidates]
        costs = [(m.prompt_cost_per_1k or 0) + (m.completion_cost_per_1k or 0) for m in candidates]
        qualities = [m.quality_score or 0.5 for m in candidates]

        max_lat = max(latencies) or 1
        max_cost = max(costs) or 1

        scores = []
        for i, model in enumerate(candidates):
            lat_score = _clamp(1.0 - latencies[i] / max_lat)
            cost_score = _clamp(1.0 - costs[i] / max_cost)
            qual_score = _clamp(qualities[i])
            composite = (
                weights.latency * lat_score
                + weights.cost * cost_score
                + weights.quality * qual_score
            )
            scores.append(ModelScore(
                model_id=model.id,
                latency_score=round(lat_score, 4),
                cost_score=round(cost_score, 4),
                quality_score=round(qual_score, 4),
                composite_score=round(composite, 4),
            ))

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        if scores:
            scores[0].selected = True
        return scores

    def route(self, request: RouteRequest) -> RoutingDecision:
        weights = self._get_weights(request)
        candidates = self._get_candidates(request)
        scores = self._score_models(candidates, weights)

        if not scores:
            # Absolute fallback
            fallback = next(
                (self.models[m] for m in self.fallback_models if m in self.models),
                list(self.models.values())[0],
            )
            return RoutingDecision(
                selected_model=fallback.id,
                weights_used=weights,
                candidate_scores=[],
                reason="fallback — no candidates after filtering",
                estimated_latency_ms=fallback.avg_latency_ms,
                estimated_cost=(fallback.prompt_cost_per_1k or 0) + (fallback.completion_cost_per_1k or 0),
            )

        best = scores[0]
        selected = self.models[best.model_id]
        reason = (
            f"Best composite score {best.composite_score:.3f} "
            f"(latency={best.latency_score:.2f}, cost={best.cost_score:.2f}, quality={best.quality_score:.2f}) "
            f"with priority='{request.priority}'"
        )

        return RoutingDecision(
            selected_model=best.model_id,
            weights_used=weights,
            candidate_scores=scores,
            reason=reason,
            estimated_latency_ms=selected.avg_latency_ms,
            estimated_cost=(selected.prompt_cost_per_1k or 0) + (selected.completion_cost_per_1k or 0),
        )

    def get_fallback(self, failed_model: str, request: RouteRequest) -> Optional[str]:
        """Return a fallback model ID when primary fails."""
        for model_id in self.fallback_models:
            if model_id != failed_model and model_id in self.models:
                return model_id
        # Last resort: any model that isn't the failed one
        for model_id in self.models:
            if model_id != failed_model:
                return model_id
        return None
