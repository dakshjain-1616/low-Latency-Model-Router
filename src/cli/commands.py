"""
CLI management tool for the model router.
"""
import asyncio
import json
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.router.core import ModelRouter, DEFAULT_MODELS
from src.router.cache import ResponseCache, MockCache
from src.router.metrics import MetricsTracker
from src.models import RouteRequest, Message, RoutingWeights

app = typer.Typer(help="Low-Latency Model Router CLI", add_completion=False)
console = Console()


def _make_router() -> ModelRouter:
    return ModelRouter()


def _make_cache(use_redis: bool = False) -> ResponseCache | MockCache:
    if use_redis:
        cache = ResponseCache()
        if cache.connect():
            return cache
        console.print("[yellow]Redis unavailable, using mock cache.[/yellow]")
    return MockCache()


@app.command("models")
def list_models():
    """List all available models with their metadata."""
    table = Table(title="Available Models", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Cost /1k tokens", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Context", justify="right")

    for m in DEFAULT_MODELS:
        cost = (m.prompt_cost_per_1k or 0) + (m.completion_cost_per_1k or 0)
        table.add_row(
            m.id,
            m.provider,
            str(m.avg_latency_ms or "N/A"),
            f"${cost:.5f}",
            f"{m.quality_score:.2f}" if m.quality_score else "N/A",
            str(m.context_length or "N/A"),
        )
    console.print(table)


@app.command("route")
def route_message(
    message: str = typer.Argument(..., help="User message to route"),
    priority: str = typer.Option("balanced", help="Priority: speed | cost | quality | balanced"),
    max_latency: Optional[int] = typer.Option(None, help="Max acceptable latency in ms"),
    max_cost: Optional[float] = typer.Option(None, help="Max cost per 1k tokens"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show routing decision without API call"),
):
    """Route a message and show the routing decision."""
    router = _make_router()
    request = RouteRequest(
        messages=[Message(role="user", content=message)],
        priority=priority,
        max_latency_ms=max_latency,
        max_cost_per_1k=max_cost,
    )
    decision = router.route(request)

    panel_content = (
        f"[bold green]Selected:[/bold green] {decision.selected_model}\n"
        f"[bold]Reason:[/bold] {decision.reason}\n"
        f"[bold]Est. Latency:[/bold] {decision.estimated_latency_ms} ms\n"
        f"[bold]Est. Cost:[/bold] ${decision.estimated_cost:.6f}/1k tokens\n"
    )
    console.print(Panel(panel_content, title="Routing Decision", border_style="blue"))

    table = Table(title="Candidate Scores", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Latency Score", justify="right")
    table.add_column("Cost Score", justify="right")
    table.add_column("Quality Score", justify="right")
    table.add_column("Composite", justify="right", style="bold")
    table.add_column("Selected", justify="center")

    for s in decision.candidate_scores:
        table.add_row(
            s.model_id,
            f"{s.latency_score:.3f}",
            f"{s.cost_score:.3f}",
            f"{s.quality_score:.3f}",
            f"{s.composite_score:.3f}",
            "[green]YES[/green]" if s.selected else "",
        )
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry-run mode — no API call made.[/yellow]")
        return

    from src.router.openrouter import OpenRouterClient, OpenRouterError
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]OPENROUTER_API_KEY not set — skipping live call.[/red]")
        return

    async def _call():
        client = OpenRouterClient()
        try:
            resp = await client.chat_completion(decision.selected_model, request, decision)
            return resp
        finally:
            await client.close()

    try:
        response = asyncio.run(_call())
        if response.choices:
            content = response.choices[0].get("message", {}).get("content", "")
            console.print(Panel(content, title=f"Response from {response.model}", border_style="green"))
        console.print(f"[dim]Latency: {response.latency_ms:.1f} ms | Cached: {response.cached}[/dim]")
    except Exception as e:
        console.print(f"[red]API call failed: {e}[/red]")


@app.command("benchmark")
def benchmark(
    iterations: int = typer.Option(5, help="Number of routing decisions to benchmark"),
    priority: str = typer.Option("balanced", help="Priority mode"),
):
    """Benchmark routing decision speed (no API calls)."""
    import time
    router = _make_router()
    request = RouteRequest(
        messages=[Message(role="user", content="What is the capital of France?")],
        priority=priority,
    )

    console.print(f"[bold]Benchmarking {iterations} routing decisions...[/bold]")
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        decision = router.route(request)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        console.print(f"  [{i+1}/{iterations}] → {decision.selected_model} in {elapsed:.3f} ms")

    avg = sum(times) / len(times)
    console.print(f"\n[green]Average routing overhead: {avg:.3f} ms[/green]")
    if avg < 50:
        console.print("[green]✓ Meets <50ms overhead requirement.[/green]")
    else:
        console.print("[red]✗ Exceeds 50ms overhead requirement.[/red]")


@app.command("cache-stats")
def cache_stats(
    use_redis: bool = typer.Option(False, "--redis", help="Connect to Redis"),
):
    """Show cache statistics."""
    cache = _make_cache(use_redis)
    stats = cache.get_stats()
    console.print_json(json.dumps(stats))


@app.command("clear-cache")
def clear_cache(
    use_redis: bool = typer.Option(False, "--redis", help="Connect to Redis"),
):
    """Clear all cached responses."""
    cache = _make_cache(use_redis)
    count = cache.invalidate()
    console.print(f"[green]Cleared {count} cache entries.[/green]")


if __name__ == "__main__":
    app()
