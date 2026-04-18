# Low-Latency Model Router

> Built autonomously by **[NEO — Your Autonomous AI Engineering Agent](https://heyneo.com)**
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-007ACC?logo=visualstudiocode&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)  [![Cursor Extension](https://img.shields.io/badge/Cursor-Install%20NEO-000000?logo=cursor&logoColor=white)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

![Models](https://img.shields.io/badge/Models-6-blue) ![Tests](https://img.shields.io/badge/Tests-29%20passing-brightgreen) ![Routing Overhead](https://img.shields.io/badge/Routing%20Overhead-%3C0.1ms-brightgreen) ![Context](https://img.shields.io/badge/Max%20Context-1M%20tokens-orange) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenRouter](https://img.shields.io/badge/Powered%20by-OpenRouter-purple)

Stop hardcoding model names. This router picks the **best LLM for each request automatically** — choosing across OpenRouter's catalogue based on your priority (speed, cost, or quality) in under 1 ms of overhead.

## Why use this?

When you call an LLM API directly you make two implicit bets: that you picked the right model, and that it stays the right choice tomorrow. In practice:

- The fastest model for one query is too slow for another.
- The cheapest model loses quality at scale.
- Models go down; you need automatic fallback.
- You're paying for repeated identical requests that could be cached.

This router solves all four. Send it any chat request, tell it your priority, and it handles the rest — routing, caching, fallback, and metrics.

## How it works

Every model in the catalogue is scored on three dimensions:

```
Score = w_latency × (1 - norm_latency)
      + w_cost    × (1 - norm_cost)
      + w_quality × quality_score
```

You control the weights via a `priority` flag:

| Priority | Latency weight | Cost weight | Quality weight | Best for |
|----------|---------------|-------------|----------------|----------|
| `speed` | 0.70 | 0.20 | 0.10 | Real-time chat, autocomplete |
| `cost` | 0.20 | 0.70 | 0.10 | High-volume batch jobs |
| `quality` | 0.10 | 0.20 | 0.70 | Summarisation, analysis |
| `balanced` | 0.40 | 0.30 | 0.30 | General use (default) |

### Model catalogue (as of April 2026)

| Model | Provider | Avg Latency | Cost /1k tokens | Quality | Context |
|-------|----------|-------------|-----------------|---------|---------|
| `google/gemini-3.1-flash-lite-preview` | Google | 400 ms | $0.00175 | 0.78 | 1M |
| `openai/gpt-5.4-mini` | OpenAI | 700 ms | $0.00525 | 0.85 | 400K |
| `anthropic/claude-sonnet-4.6` | Anthropic | 900 ms | $0.018 | 0.92 | 1M |
| `google/gemini-3.1-pro-preview` | Google | 1100 ms | $0.014 | 0.93 | 1M |
| `openai/gpt-5.4` | OpenAI | 1200 ms | $0.0175 | 0.96 | 1M |
| `anthropic/claude-opus-4.7` | Anthropic | 1500 ms | $0.030 | 0.98 | 1M |

If the selected model fails, the router automatically retries with the next-best candidate. Identical requests are served from cache (Redis, or in-memory if Redis is unavailable).

## Features

- Routing decision overhead **< 1 ms** (no network call)
- Automatic **fallback** on model failure
- **Redis caching** keyed by request hash — falls back to in-memory automatically
- Rolling-window **metrics** (avg / p95 / p99 latency, per-model usage, cache hit rate)
- **FastAPI REST server** with interactive docs at `/docs`
- **CLI** for exploration, dry-runs, and benchmarking without writing code

## Project Structure

```
ml_project_0652/
├── src/
│   ├── models.py              # Pydantic schemas
│   ├── router/
│   │   ├── core.py            # Weighted scoring engine + model catalogue
│   │   ├── metrics.py         # Rolling-window metrics tracker
│   │   ├── openrouter.py      # Async OpenRouter API client
│   │   └── cache.py           # Redis cache + MockCache fallback
│   ├── api/
│   │   ├── main.py            # FastAPI app
│   │   └── routes.py          # Route definitions
│   └── cli/
│       └── commands.py        # Typer CLI
├── tests/                     # 29 unit + integration tests
├── start_router.py            # Server entry point
├── config.yaml                # Server, Redis, and routing config
├── .env.example               # Environment variable template
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key

```bash
cp .env.example .env
# Edit .env and replace the placeholder:
# OPENROUTER_API_KEY=sk-or-your-key-here
```

Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys).  
The key is only needed for live API calls — dry-runs, CLI exploration, and tests all work without it.

### 3. (Optional) Start Redis for persistent caching

```bash
# Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or system package
redis-server
```

If Redis is not running the router falls back to an in-memory cache automatically — no configuration needed.

### 4. Start the server

```bash
python start_router.py
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

---

## REST API

### Route a request (balanced — default)

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "priority": "balanced"
  }'
```

**Response:**
```json
{
  "id": "gen-abc123",
  "model": "google/gemini-flash-1.5",
  "choices": [{"message": {"role": "assistant", "content": "Paris."}}],
  "usage": {"prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17},
  "routing_decision": {
    "selected_model": "google/gemini-flash-1.5",
    "reason": "Best composite score 0.768 (latency=0.58, cost=0.98, quality=0.80) with priority='balanced'"
  },
  "latency_ms": 312.4,
  "cached": false
}
```

### Route with speed priority and a latency cap

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Translate: hello"}],
    "priority": "speed",
    "max_latency_ms": 700
  }'
```

### All endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/route` | Route a chat completion to the optimal model |
| `GET` | `/models` | List all models with latency, cost, and quality metadata |
| `GET` | `/metrics` | Rolling-window stats (avg/p95/p99 latency, model usage) |
| `GET` | `/health` | Redis + OpenRouter connectivity check |
| `GET` | `/cache/stats` | Cache entry count and memory usage |
| `DELETE` | `/cache` | Invalidate all cached responses |

---

## CLI

All CLI commands run from the project root (`ml_project_0652/`).

### List available models

```bash
python -m src.cli.commands models
```

### Preview a routing decision without making an API call

```bash
python -m src.cli.commands route "What is 2+2?" --dry-run
```

### Route with quality priority (dry-run)

```bash
python -m src.cli.commands route "Summarize this article" --priority quality --dry-run
```

### Route with a latency cap (dry-run)

```bash
python -m src.cli.commands route "Hello" --priority speed --max-latency 600 --dry-run
```

### Make a live API call (requires OPENROUTER_API_KEY in .env)

```bash
python -m src.cli.commands route "What is 2+2?" --priority balanced
```

### Benchmark routing decision speed

```bash
python -m src.cli.commands benchmark --iterations 10
```

Expected output: routing overhead well under 1 ms per decision.

### Cache management

```bash
# Show cache stats (in-memory by default)
python -m src.cli.commands cache-stats

# Show Redis cache stats
python -m src.cli.commands cache-stats --redis

# Clear in-memory cache
python -m src.cli.commands clear-cache

# Clear Redis cache
python -m src.cli.commands clear-cache --redis
```

---

## Tests

```bash
# Run all 29 tests
python -m pytest tests/ -v

# By module
python -m pytest tests/test_router_core.py -v   # routing engine
python -m pytest tests/test_cache.py -v         # cache layer
python -m pytest tests/test_metrics.py -v       # metrics tracker
python -m pytest tests/test_api.py -v           # FastAPI endpoints (mocked)
```

All API tests use mocked OpenRouter responses — no API key required.

---

## Configuration

Edit `config.yaml` to change defaults:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

redis:
  host: "localhost"
  port: 6379
  ttl_seconds: 3600        # How long to cache responses

routing:
  default_weights:
    latency: 0.4
    cost: 0.3
    quality: 0.3
  fallback_models:          # Tried in order when primary fails
    - "openai/gpt-4o-mini"
    - "anthropic/claude-3-haiku"
    - "google/gemini-flash-1.5"
```

---

## Requirements

- Python 3.10+
- `OPENROUTER_API_KEY` — only needed for live inference (not for tests or `--dry-run`)
- Redis — optional; in-memory fallback is automatic
