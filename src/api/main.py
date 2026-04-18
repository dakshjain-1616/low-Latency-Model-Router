"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.router.core import ModelRouter
from src.router.cache import ResponseCache, MockCache
from src.router.metrics import MetricsTracker


router_instance = ModelRouter()
cache_instance: ResponseCache | MockCache = MockCache()
metrics_instance = MetricsTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache_instance
    real_cache = ResponseCache()
    if real_cache.connect():
        cache_instance = real_cache
        print("Connected to Redis.")
    else:
        print("Redis unavailable — using in-memory mock cache.")
    yield
    if hasattr(cache_instance, "disconnect"):
        cache_instance.disconnect()


app = FastAPI(
    title="Low-Latency Model Router",
    description="Production-ready LLM router with intelligent model selection, Redis caching, and metrics.",
    version="1.0.0",
    lifespan=lifespan,
)

from src.api.routes import router as api_router  # noqa: E402
app.include_router(api_router)
