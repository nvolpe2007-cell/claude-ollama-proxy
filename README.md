# claude-ollama-proxy

[![Test](https://github.com/nvolpe2007-cell/claude-ollama-proxy/actions/workflows/test.yml/badge.svg)](https://github.com/nvolpe2007-cell/claude-ollama-proxy/actions/workflows/test.yml)

A lightweight Node.js proxy that translates Anthropic API requests (Claude format) into Ollama's OpenAI-compatible format. This lets you point **Claude Code** — or any tool that speaks the Anthropic `messages` API — at a local Ollama model instead of the real Claude API.

Also accepts native **OpenAI chat format** directly, making it a drop-in proxy for OpenAI-compatible clients (Cursor, Continue, LiteLLM, etc.).

## How it works

```
Claude Code  →  POST /v1/messages (Anthropic format)
               ↓  proxy translates
            Ollama /v1/chat/completions (OpenAI format)
               ↓  proxy translates back
Claude Code  ←  Anthropic response (streaming SSE or JSON)
```

The proxy handles:
- Full Anthropic `messages` format including multi-part content blocks
- Tool use / tool results (`tool_use`, `tool_result`) ↔ OpenAI function calling, including images returned in `tool_result` blocks
- Image content blocks (base64 and URL sources) → OpenAI vision format
- `document` content blocks → plain text for Ollama
- Both streaming (SSE) and non-streaming responses
- `<think>…</think>` reasoning blocks ↔ Anthropic `thinking` content blocks, including interleaved thinking
- Graceful errors when Ollama is offline
- Native OpenAI `/v1/chat/completions` and `/v1/completions` passthrough
- OpenAI `/v1/embeddings`
- Anthropic Messages Batch API (`/v1/messages/batches*`)
- Model management (list, inspect, pull, delete) via `/v1/models*`

## Requirements

- [Node.js](https://nodejs.org/) v18 or newer (uses native `fetch`)
- [Ollama](https://ollama.com/) running locally with at least one model pulled

## Install

```bash
git clone https://github.com/nvolpe2007-cell/claude-ollama-proxy.git
cd claude-ollama-proxy
# No npm install needed — zero dependencies
```

## Run

```bash
node proxy.js
# or
npm start
```

You should see:

```
  Claude-Ollama proxy ready
  Model : qwen2.5:7b
  Port  : 4000
  Bind  : 0.0.0.0 (all interfaces — set PROXY_LISTEN_HOST=127.0.0.1 to restrict)
  Ollama: http://localhost:11434
```

The proxy also loads a `.env` file from the current working directory (or its own directory) at startup. Shell environment variables always take precedence over `.env` values.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Default Ollama model used when the client doesn't specify one (or sends a `claude-*` name with no `MODEL_MAP` match) |
| `PROXY_PORT` | `4000` | Port the proxy listens on |
| `PROXY_LISTEN_HOST` | all interfaces | Address to bind to. Set to `127.0.0.1` to restrict the proxy to localhost-only access (no LAN/WAN exposure without a firewall) |
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of your Ollama instance. Accepts a comma-separated list of URLs for round-robin load distribution across multiple Ollama instances/GPUs |
| `PROXY_API_KEY` | *(unset)* | If set, require this key on every API request (`x-api-key` or `Authorization: Bearer`). Tracked under the key name `default` in `/metrics` |
| `PROXY_API_KEYS` | *(unset)* | Comma-separated list of additional named keys for multi-caller setups, e.g. `nick:sk-abc,family:sk-def`. Combines with `PROXY_API_KEY`. Bare entries without a `name:` prefix are auto-named `key1`, `key2`, ... Each key's usage is tracked separately in `/metrics` and the dashboard |
| `PROXY_API_KEY_MODELS` | *(unset)* | Per-key model allow-list, e.g. `family:llama3.2:1b,kids:llama3.2:1b\|qwen2.5:7b`. Restricts the named key (matching a `PROXY_API_KEYS`/`PROXY_API_KEY` name, or `default`) to only the listed Ollama models; multiple models for one key are `\|`-separated. Keys with no entry here are unrestricted. Returns `403 permission_error` on disallowed models |
| `MODEL_MAP` | *(unset)* | JSON map of `claude-*` names/prefixes to Ollama models (see below) |
| `PROXY_TLS_CERT` | *(unset)* | Path to PEM certificate file — enables HTTPS when set |
| `PROXY_TLS_KEY` | *(unset)* | Path to PEM private key file — required when cert is set |
| `CORS_ORIGIN` | `*` | Value for `Access-Control-Allow-Origin`; set to a specific origin to restrict browser access |
| `OLLAMA_NUM_CTX` | *(model default)* | Context window size sent to Ollama. Model defaults are often only 2048 — set to `32768` or higher for real sessions |
| `OLLAMA_KEEP_ALIVE` | *(Ollama default)* | How long the model stays loaded in GPU memory between requests (`"30m"`, `"0"` to unload immediately, `"-1"` to keep forever) |
| `OLLAMA_OPTIONS` | *(unset)* | JSON object of arbitrary Ollama model parameters applied to every request (`repeat_penalty`, `mirostat`, `num_gpu`, `tfs_z`, `typical_p`, etc). Per-request client params take precedence; `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` take highest precedence |
| `PROXY_TIMEOUT` | *(none)* | Hard per-request timeout in milliseconds. If Ollama does not respond within this window the proxy aborts and returns a `504` error (non-streaming) or an SSE error event (streaming) |
| `PROXY_IDLE_TIMEOUT` | *(none)* | Idle stream timeout in milliseconds. Aborts a streaming response if no new tokens arrive from Ollama for this long — catches a model that's stalled mid-generation without needing a high `PROXY_TIMEOUT` |
| `PROXY_MAX_TOKENS` | `8192` | Default `max_tokens` applied when the client omits the field |
| `PROXY_HARD_MAX_TOKENS` | *(none)* | Hard ceiling on output tokens per request. Any client-requested `max_tokens` above this is silently clamped down; also caps the `PROXY_MAX_TOKENS` default. Useful on shared deployments to stop one caller from monopolising the GPU |
| `PROXY_MAX_BODY_SIZE` | *(none)* | Hard limit on request body size in bytes. Requests that exceed this are rejected with `413`. Protects against runaway base64-image payloads. Example: `10485760` for 10 MB |
| `PROXY_SYSTEM_PROMPT` | *(unset)* | Operator-defined text prepended to every request's system prompt. When the client also supplies a system prompt, the proxy's text comes first, separated by two newlines |
| `PROXY_WARMUP` | `false` | When `true`, sends a minimal preflight request to Ollama after startup to pre-load the configured model into GPU memory (all hosts in `OLLAMA_HOST`, in parallel), eliminating cold-start latency on the first real request |
| `PROXY_FORCE_THINK` | `false` | When `true`, unconditionally adds `think:true` to every outgoing Ollama request, enabling chain-of-thought reasoning for thinking models (DeepSeek-R1, Qwen3-thinking) without the client opting in. Safe for non-thinking models — Ollama ignores the field |
| `PROXY_AUTO_TRUNCATE` | `false` | When `true` and `OLLAMA_NUM_CTX` is set, automatically drops the oldest user/assistant turns when the estimated input would exceed `OLLAMA_NUM_CTX`, preventing "context length exceeded" errors in long sessions. Sets `x-context-truncated` response header when it triggers |
| `RATE_LIMIT_RPM` | *(none)* | Global request rate limit in requests per minute across all callers. Applies to `POST /v1/messages`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages/count_tokens`, `/v1/embeddings`, `POST /v1/messages/batches`, all batch sub-routes (`GET /v1/messages/batches*`, `POST .../cancel`, `DELETE .../batches/:id`), `/v1/models/pull`, `DELETE /v1/models/:id`, `GET /v1/models`, and `GET /v1/models/:id`. Returns `429` with `retry-after` header when exceeded |
| `RATE_LIMIT_PER_IP_RPM` | *(none)* | Per-client-IP rate limit in requests per minute. Buckets by the raw socket address unless `PROXY_TRUST_PROXY=true` |
| `RATE_LIMIT_PER_KEY_RPM` | *(none)* | Per-API-key rate limit in requests per minute. Buckets by the caller's matched `PROXY_API_KEYS` name (or `"default"` when no API keys are configured). Lets each device/user in a multi-key deployment have its own budget regardless of shared IPs. All three rate limits can be active simultaneously |
| `PROXY_TRUST_PROXY` | `false` | When `true`, `RATE_LIMIT_PER_IP_RPM` reads the client IP from `x-forwarded-for` instead of the raw socket address. Only enable this when the proxy sits behind a reverse proxy/load balancer that overwrites the header with the real client IP — otherwise any direct caller can spoof a fresh IP on every request and bypass the per-IP limit entirely |
| `PROXY_MAX_CONCURRENCY` | *(none)* | Maximum simultaneous in-flight Ollama inference requests. When reached, new requests get `503 overloaded_error` (or queue — see `PROXY_MAX_QUEUE_SIZE`). Prevents GPU VRAM OOM on single-GPU setups |
| `PROXY_MAX_QUEUE_SIZE` | *(none)* | Number of requests that may wait in a queue when all concurrency slots are taken, instead of immediately returning `503`. Pairs with `PROXY_MAX_CONCURRENCY` |
| `PROXY_MAX_QUEUE_TIMEOUT` | *(none)* | Milliseconds a queued request waits before giving up with `503`. Only meaningful with `PROXY_MAX_QUEUE_SIZE` |
| `LOG_FORMAT` | `text` | Log format for request lines. `text` writes human-readable lines; `json` writes a single JSON object per request — useful for log aggregation tools like Grafana Loki, Datadog, or AWS CloudWatch |
| `LOG_LEVEL` | `info` | `info` logs one summary line per request. `debug` additionally logs the full translated OpenAI-format request sent to Ollama and the raw Ollama response (non-streaming) — invaluable for diagnosing message/tool conversion issues. Large base64 image payloads are truncated in debug logs |
| `PROXY_BATCH_PERSIST_PATH` | *(unset)* | Path to a JSON file where Messages Batch API state (requests + results) is saved after every change and reloaded at startup. Without this, batches and their results live only in memory and are lost on restart. Batches that hadn't finished are resumed, skipping any items already completed |

Examples:

```bash
OLLAMA_MODEL=llama3.1:8b node proxy.js
OLLAMA_HOST=http://192.168.1.50:11434 node proxy.js   # remote Ollama instance
PROXY_API_KEY=mysecret node proxy.js                  # enable auth
OLLAMA_NUM_CTX=32768 PROXY_WARMUP=true node proxy.js  # production setup
PROXY_MAX_CONCURRENCY=1 PROXY_MAX_QUEUE_SIZE=10 node proxy.js  # single-GPU, queue bursts
```

## Authentication

By default the proxy is open to anyone who can reach port 4000. To restrict access, set `PROXY_API_KEY` to any secret string:

```bash
PROXY_API_KEY=mysecret node proxy.js
```

Callers must then pass the key in one of two ways:

```bash
# Option A — x-api-key header (what Claude Code uses by default)
curl http://localhost:4000/v1/messages \
  -H 'x-api-key: mysecret' \
  ...

# Option B — Authorization Bearer
curl http://localhost:4000/v1/messages \
  -H 'Authorization: Bearer mysecret' \
  ...
```

When using Claude Code, set `ANTHROPIC_API_KEY=mysecret` and it will automatically send it as `x-api-key`.

### Multiple API keys (per-device / per-user)

If you share the proxy with other devices, family members, or apps, set `PROXY_API_KEYS` to a comma-separated list of `name:key` pairs so each caller gets its own key:

```bash
PROXY_API_KEYS="nick-laptop:sk-abc123,family-pc:sk-def456,homeassistant:sk-ghi789" node proxy.js
```

This combines with `PROXY_API_KEY` if both are set (the latter is tracked under the name `default`). Bare entries without a `name:` prefix (e.g. `PROXY_API_KEYS=sk-abc123,sk-def456`) are auto-named `key1`, `key2`, etc.

Each caller authenticates the same way (`x-api-key` or `Authorization: Bearer`), but with their own key. Per-key request counts and token usage are broken out separately in `GET /metrics` (`api_keys_usage`), `GET /metrics/prometheus` (`proxy_api_key_requests_total` / `proxy_api_key_tokens_total`), and the live dashboard — so you can see who's using how much, and revoke a single caller's key (by removing it from `PROXY_API_KEYS` and restarting) without rotating everyone else's.

The `/health` and `/metrics` endpoints are always unauthenticated so monitoring tools can reach them freely.

### Per-key model access control

By default any caller can request any Ollama model. To restrict which models a given key can use, set `PROXY_API_KEY_MODELS` to a comma-separated list of `name:model1|model2|...` entries (the name matches a `PROXY_API_KEYS`/`PROXY_API_KEY` name, or `default` when no named keys are configured):

```bash
PROXY_API_KEYS="nick:sk-abc123,family:sk-def456" \
PROXY_API_KEY_MODELS="family:llama3.2:1b" \
node proxy.js
```

Here `nick`'s key has unrestricted access, while `family`'s key can only use `llama3.2:1b` — any other model (including `claude-*` aliases that resolve to a different Ollama model) gets a `403 permission_error`. Use `|` to allow multiple models for one key, e.g. `kids:llama3.2:1b|qwen2.5:7b`. Keys with no entry in `PROXY_API_KEY_MODELS` are unaffected.

The check applies to `POST /v1/messages`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages/count_tokens`, and each item in `POST /v1/messages/batches` — after `MODEL_MAP` alias resolution, so the restriction is enforced against the actual Ollama model name. It also applies to `GET /v1/models`/`GET /v1/models/:id` (restricted keys only see models they're allowed to use) and to `DELETE /v1/models/:id` / `POST /v1/models/pull`, so a restricted key can't delete or download models outside its allow-list either.

`GET /v1/models` and `GET /v1/models/:id` respect the same allow-list: a restricted key only sees the Ollama models (and matching `MODEL_MAP` aliases) it's permitted to use, and looking up a model outside its allow-list returns `404 not_found_error` — so model-picker UIs (Continue, Open WebUI, Cursor) never offer a choice that would 403.

## Point Claude Code at the proxy

Claude Code reads the `ANTHROPIC_BASE_URL` environment variable. Set it before starting Claude Code:

```bash
# In the same terminal session as Claude Code:
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=ollama   # must be set but value is ignored by the proxy

claude
```

Or add both to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make it permanent.

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:4000"
$env:ANTHROPIC_API_KEY  = "ollama"
claude
```

## Models

### List available models

```bash
curl http://localhost:4000/v1/models
```

Returns the models currently loaded in Ollama in OpenAI-compatible format, including `details` (family, parameter size, quantization) and `size` from Ollama's `/api/tags`:

```json
{
  "object": "list",
  "data": [
    { "id": "qwen2.5:7b", "object": "model", "owned_by": "ollama", "size": 4683087519,
      "details": { "family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M" } }
  ]
}
```

When `MODEL_MAP` is configured, Claude alias names (e.g. `claude-3-haiku`) are also included in the list so model-picker clients like Cursor and Continue can discover and select them.

If `PROXY_API_KEY_MODELS` is configured, the list (and `GET /v1/models/:id`) is filtered per-caller — see [Per-key model access control](#per-key-model-access-control).

### Look up, pull, and delete models

```bash
# Look up a single model — also enriches with context_length, system, and template
curl http://localhost:4000/v1/models/qwen2.5:7b

# Pull a new model from the Ollama registry through the proxy (auth-gated)
curl -X POST http://localhost:4000/v1/models/pull \
  -H 'x-api-key: mysecret' -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder:7b"}'

# Pull with streaming progress (SSE)
curl -N -X POST http://localhost:4000/v1/models/pull \
  -H 'x-api-key: mysecret' -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-coder:7b","stream":true}'

# Delete a model from Ollama (auth-gated)
curl -X DELETE http://localhost:4000/v1/models/qwen2.5-coder:7b -H 'x-api-key: mysecret'
```

This lets you manage your model library without SSH access to the Ollama host.

## Live dashboard

Open **http://localhost:4000/** in your browser to see a live status dashboard. It auto-refreshes every 5 seconds and shows:

- Ollama connectivity status and per-host health (for multi-host deployments)
- Active streaming connections, in-flight/queued LLM requests (with `PROXY_MAX_CONCURRENCY`/`PROXY_MAX_QUEUE_SIZE`)
- Startup config (model, port, auth, TLS, rate limits, context window, concurrency limits, etc.)
- Per-route request counts and HTTP status code breakdown
- p50/p95/p99/min/max/avg latency
- Cumulative input and output token totals
- Per-model usage breakdown
- Per-API-key usage breakdown (multi-key deployments)

No external dependencies — the dashboard is served directly by the proxy and polls its own `/health` and `/metrics` endpoints.

## Check the proxy is running

```bash
curl http://localhost:4000/health
```

Returns `200 OK` when at least one Ollama host is reachable:

```json
{
  "status": "ok",
  "proxy": "running",
  "version": "1.0.0",
  "hosts": [{ "url": "http://localhost:11434", "status": "ok" }],
  "ollama": "reachable",
  "model": "qwen2.5:7b",
  "model_available": true,
  "port": 4000,
  "timestamp": "2026-04-27T12:00:00.000Z"
}
```

Returns `503` with `"status": "degraded"` if any configured Ollama host is unreachable (`"status": "ok"` only when *all* hosts are reachable).

### Model availability check

Every `/health` call also checks whether the configured model(s) have actually been pulled into Ollama, by cross-referencing `OLLAMA_MODEL` (and every `MODEL_MAP` target) against each reachable host's `/api/tags` model list:

- `model_available` — `true`/`false` for `OLLAMA_MODEL`, or `null` if no host could be checked (e.g. all hosts unreachable)
- `models_status` — present when more than one distinct model is configured (via `MODEL_MAP`); maps each model name to `true`/`false`
- `warning` — present when one or more configured models are missing, e.g. `"Models not found on any reachable Ollama host: qwen2.5:14b — run 'ollama pull <model>'"`

If any configured model is missing, `status` becomes `"degraded"` even though Ollama itself is reachable — this catches the common misconfiguration of pointing `OLLAMA_MODEL`/`MODEL_MAP` at a model that hasn't been pulled yet, which otherwise only surfaces as a confusing error on the first real request. The live dashboard (`GET /`) also shows a "Model — not pulled ⚠" warning in the Status card when this happens.

## Metrics

### JSON metrics

```bash
curl http://localhost:4000/metrics
```

Returns an in-memory snapshot including:
- Uptime, total request counts per route, HTTP status code breakdown
- p50/p95/p99 latency percentiles (rolling 1000-sample window), min/avg/max
- Cumulative input and output token totals
- Per-model breakdown of request counts and token usage (`models_usage`)
- Per-API-key breakdown of request counts and token usage (`api_keys_usage`)
- Current active streaming connection count, active/queued LLM request counts, and configured concurrency/queue limits

### Prometheus metrics

```bash
curl http://localhost:4000/metrics/prometheus
```

Returns the same data in [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) (version 0.0.4), ready to be scraped directly by Prometheus without any additional exporter. Useful for Grafana dashboards.

Exposed metrics include `proxy_requests_total`, `proxy_http_responses_total`, `proxy_request_duration_ms`, `proxy_tokens_total`, `proxy_model_requests_total`, `proxy_model_tokens_total`, `proxy_api_key_requests_total`, `proxy_api_key_tokens_total`, `proxy_active_streams`, `proxy_active_llm_requests`, `proxy_queued_llm_requests`, `proxy_concurrency_limit`, `proxy_queue_limit`, and `proxy_errors_total`.

## OpenAI-compatible passthrough

The proxy also accepts native OpenAI chat format directly, making it a drop-in for OpenAI-compatible clients (Cursor, Continue, LiteLLM, etc.):

```bash
POST /v1/chat/completions
```

Requests are forwarded to Ollama with no format translation. All proxy features apply: authentication, rate limiting, retry, timeout, client-abort propagation, keepalive SSE comments, request logging, and per-model metrics. `MODEL_MAP` and `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` tuning are applied. Both streaming and non-streaming are supported. `max_completion_tokens` is accepted as an alias for `max_tokens`.

### Legacy text completions

```bash
POST /v1/completions
```

Accepts the older OpenAI text completions format (`prompt` as a string or array, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `stream`). The `prompt` is converted into a single user message and forwarded to Ollama's chat endpoint; the response is converted back into the `text_completion` envelope (`choices[].text`). Streaming emits SSE deltas in completions format. Useful for older tools and LiteLLM configs that still target `/v1/completions`.

## Embeddings

```bash
curl http://localhost:4000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text","input":"Hello world"}'
```

Forwards to Ollama's `/api/embed` and returns an OpenAI-compatible envelope:

```json
{
  "object": "list",
  "data": [{ "object": "embedding", "embedding": [0.01, -0.02, ...], "index": 0 }],
  "model": "nomic-embed-text",
  "usage": { "prompt_tokens": 2, "total_tokens": 2 }
}
```

`input` may be a string or an array of strings. Supports any Ollama embedding model (`nomic-embed-text`, `mxbai-embed-large`, etc.) and resolves `MODEL_MAP` aliases. Auth, rate limiting, timeout, retry, and `OLLAMA_OPTIONS` all apply.

`encoding_format` is also supported: `"float"` (default) returns the plain JSON array shown above; `"base64"` returns each `embedding` as a base64 string of little-endian float32 bytes, matching the real OpenAI API's wire format — needed by clients that request it explicitly, such as LangChain's `OpenAIEmbeddings`, which defaults to `encoding_format:"base64"`.

## Anthropic Messages Batch API

The proxy implements Anthropic's [Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches) backed by an in-memory queue, processed serially against Ollama:

```bash
# Create a batch
curl -X POST http://localhost:4000/v1/messages/batches \
  -H 'Content-Type: application/json' -H 'x-api-key: mysecret' \
  -d '{"requests":[
        {"custom_id":"req-1","params":{"model":"claude-3-haiku","max_tokens":100,
          "messages":[{"role":"user","content":"Say hi"}]}}
      ]}'

# Poll status
curl http://localhost:4000/v1/messages/batches/msgbatch_xxx -H 'x-api-key: mysecret'

# List all batches
curl http://localhost:4000/v1/messages/batches -H 'x-api-key: mysecret'

# Stream JSONL results once processing_status is "ended"
curl http://localhost:4000/v1/messages/batches/msgbatch_xxx/results -H 'x-api-key: mysecret'

# Cancel a batch
curl -X POST http://localhost:4000/v1/messages/batches/msgbatch_xxx/cancel -H 'x-api-key: mysecret'

# Delete an ended batch
curl -X DELETE http://localhost:4000/v1/messages/batches/msgbatch_xxx -H 'x-api-key: mysecret'
```

Notes:
- A batch may contain at most 100,000 requests (matching the real Anthropic API's limit); a larger `requests` array is rejected with `400 invalid_request_error` at creation time.
- Batches expire 24 hours after creation; unprocessed items are marked `expired`. Ended batches are removed from memory after 1 hour.
- Each batch item respects `PROXY_MAX_CONCURRENCY` and competes fairly with real-time requests.
- `DELETE /v1/messages/batches/{id}` only succeeds once a batch has `processing_status: "ended"` (matching the real Anthropic API); an in-progress or canceling batch must be canceled first and allowed to finish before it can be deleted. Returns `{"id":"msgbatch_xxx","type":"message_batch_deleted"}`.
- Batches are **in-memory** by default — set `PROXY_BATCH_PERSIST_PATH` to persist batches and results across restarts.

## Rate limiting

Three independent rate limits can be set simultaneously. All apply to `POST /v1/messages`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages/count_tokens`, `/v1/embeddings`, `/v1/messages/batches` (batch creation — capped here too since a single batch can enqueue many inference requests against the same shared Ollama/GPU resource), and `/v1/models/pull` (a model pull can download many gigabytes from the registry, so it gets the same caps even though it doesn't touch the GPU).

```bash
# Global cap: 60 requests/min across all callers
RATE_LIMIT_RPM=60 node proxy.js

# Per-IP cap: 10 requests/min per client
RATE_LIMIT_PER_IP_RPM=10 node proxy.js

# Per-API-key cap: 20 requests/min per caller (see PROXY_API_KEYS)
RATE_LIMIT_PER_KEY_RPM=20 node proxy.js

# All three active at once
RATE_LIMIT_RPM=100 RATE_LIMIT_PER_IP_RPM=20 RATE_LIMIT_PER_KEY_RPM=20 node proxy.js
```

`RATE_LIMIT_PER_KEY_RPM` buckets requests by the caller's matched `PROXY_API_KEYS` name (falling back to a shared `"default"` bucket when no API keys are configured, or when `PROXY_API_KEY` is used). This is the right knob for multi-caller deployments — e.g. give `nick` and `family` each their own 20 req/min budget even though they may share an IP behind the same router.

When a limit is exceeded the proxy responds with `429 rate_limit_error` and sets `retry-after`, `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, and `x-ratelimit-reset-requests` headers — matching Anthropic's own API header naming.

## Concurrency limiting & request queueing

On a single GPU, multiple simultaneous Ollama requests can cause VRAM OOM errors or severe slowdowns. `PROXY_MAX_CONCURRENCY` caps how many inference requests (`/v1/messages`, `/v1/chat/completions`, `/v1/completions`, and batch items) run at once:

```bash
# Allow only 1 request at a time; reject extras immediately with 503
PROXY_MAX_CONCURRENCY=1 node proxy.js
```

To absorb bursts instead of immediately rejecting, add a queue:

```bash
# Allow 1 in-flight request, queue up to 10 more (each waits up to 30s)
PROXY_MAX_CONCURRENCY=1 PROXY_MAX_QUEUE_SIZE=10 PROXY_MAX_QUEUE_TIMEOUT=30000 node proxy.js
```

Queued requests that disconnect before their turn are removed without consuming a slot. Current in-flight/queued counts are shown in `/metrics` (`active_llm_requests`, `queued_llm_requests`) and the live dashboard.

## System prompt injection

Prepend a fixed system prompt to every request without modifying clients:

```bash
PROXY_SYSTEM_PROMPT="You are a helpful assistant. Always respond in English." node proxy.js
```

When the client already sends a system prompt, the proxy's text comes first, separated by two newlines. Handles string, array-of-blocks, and absent system prompts. Applied to `POST /v1/messages`, `POST /v1/chat/completions`, and `POST /v1/messages/count_tokens` so token estimates reflect the injected text.

## Multi-host round-robin

Distribute load across multiple Ollama instances by setting `OLLAMA_HOST` to a comma-separated list:

```bash
OLLAMA_HOST=http://gpu1:11434,http://gpu2:11434,http://gpu3:11434 node proxy.js
```

Requests are distributed round-robin across all listed hosts. `GET /health` checks all hosts in parallel and reports per-host status in a `hosts` array. `PROXY_WARMUP=true` pre-loads the model on every host in parallel.

### Automatic failover

The proxy tracks each host's health and routes around one that's down. After 2 consecutive failed checks a host is marked unhealthy and skipped by round-robin (so a crashed Ollama instance or unplugged GPU box doesn't keep eating every Nth request); a single successful check immediately restores it. In multi-host setups a background check pings every host's `/api/tags` every 15 seconds, and `GET /health` performs a live check on every call. Each entry in the `hosts` array includes a `routing` field — `"active"` or `"skipped"` — showing whether that host is currently eligible for selection. If every host is unhealthy, the proxy fails open and keeps rotating normally so existing per-request retry/error handling still applies.

## Model mapping (MODEL_MAP)

Map `claude-*` model names to specific Ollama models so Claude Code's model selection works naturally:

```bash
MODEL_MAP='{"claude-3-haiku":"qwen2.5:7b","claude-3-sonnet":"qwen2.5:14b","claude-3-opus":"qwen2.5:72b"}' node proxy.js
```

Exact match wins; if no exact match, any key that is a **prefix** of the requested model name is used (e.g. `"claude-3-haiku"` matches `claude-3-haiku-20240307`). Non-`claude-*` model names always pass through unchanged.

## TLS / HTTPS

Point the proxy at your PEM files to enable HTTPS:

```bash
PROXY_TLS_CERT=/etc/ssl/proxy.crt PROXY_TLS_KEY=/etc/ssl/proxy.key node proxy.js
```

## Network binding

By default the proxy listens on all interfaces (`0.0.0.0`), so it's reachable from your LAN. To restrict it to the local machine only:

```bash
PROXY_LISTEN_HOST=127.0.0.1 node proxy.js
```

Useful when running on a shared or internet-facing machine without a firewall, or when only the local user (or a reverse proxy on the same host) should connect.

## Thinking models

The proxy automatically extracts `<think>…</think>` blocks produced by reasoning models (DeepSeek-R1, Qwen3-thinking, etc.) and converts them to proper Anthropic `thinking` content blocks, including interleaved thinking (multiple think/text pairs in one response). Claude Code displays them as structured reasoning without any extra configuration.

When Claude Code enables extended thinking (sends `thinking: {type: "enabled", budget_tokens: N}`), the proxy forwards `think: true` to Ollama (Ollama 0.7+), which activates native thinking for supported models.

To enable thinking for **every** request regardless of what the client sends:

```bash
PROXY_FORCE_THINK=true OLLAMA_MODEL=qwen3:8b node proxy.js
```

Thinking blocks in assistant messages are also preserved in multi-turn conversation history — they are re-encoded as `<think>…</think>` tags when the conversation history is sent back to Ollama, so reasoning context is maintained across turns.

```bash
ollama pull deepseek-r1:7b
OLLAMA_MODEL=deepseek-r1:7b node proxy.js
```

## Recommended models

Smaller models that work well with Claude Code's tool-use patterns:

| Model | Pull command | Notes |
|---|---|---|
| `qwen2.5:7b` | `ollama pull qwen2.5:7b` | Default; good balance of speed and quality |
| `qwen2.5-coder:7b` | `ollama pull qwen2.5-coder:7b` | Better for coding tasks |
| `llama3.1:8b` | `ollama pull llama3.1:8b` | Good general purpose |
| `mistral:7b` | `ollama pull mistral:7b` | Fast, lower memory |
| `deepseek-r1:7b` | `ollama pull deepseek-r1:7b` | Thinking model; reasoning shown as structured blocks |
| `qwen3:8b` | `ollama pull qwen3:8b` | Thinking model with strong coding ability |

## Request timeouts

```bash
# Abort if Ollama hasn't responded at all within 5 minutes
PROXY_TIMEOUT=300000 node proxy.js

# Abort a stream if no new tokens arrive for 60 seconds (model stuck)
PROXY_IDLE_TIMEOUT=60000 node proxy.js
```

`PROXY_TIMEOUT` bounds total request time; `PROXY_IDLE_TIMEOUT` only fires during active streaming and resets on every chunk received, so normal slow-but-steady generation is unaffected. Both can be set together. Timeouts return `504` (non-streaming) or a `request_timeout` SSE error event (streaming), and immediately abort the in-flight Ollama request so GPU resources are freed.

## Output token limits

```bash
# Default max_tokens when the client omits it (default 8192)
PROXY_MAX_TOKENS=16384 node proxy.js

# Hard ceiling — clamps any client-requested max_tokens above this value
PROXY_HARD_MAX_TOKENS=4096 node proxy.js
```

`PROXY_HARD_MAX_TOKENS` also caps the `PROXY_MAX_TOKENS` default, so on a shared deployment you only need to set one value to stop a single caller from requesting a huge generation budget. Applies to `/v1/messages`, `/v1/chat/completions`, and `/v1/completions`.

## Ollama model options (OLLAMA_OPTIONS)

Pass arbitrary Ollama generation parameters on every request without adding a dedicated env var for each one:

```bash
OLLAMA_OPTIONS='{"repeat_penalty":1.1,"mirostat":2,"num_gpu":33}' node proxy.js
```

Per-request client values take precedence over `OLLAMA_OPTIONS`, and `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` always take highest precedence. Applies to `/v1/messages`, `/v1/chat/completions`, and `/v1/embeddings`.

## Debug logging

```bash
LOG_LEVEL=debug node proxy.js
```

Logs the full translated OpenAI-format request body sent to Ollama (`[DEBUG] → Ollama`) and, for non-streaming requests, the raw Ollama response (`[DEBUG] ← Ollama response`). Invaluable for diagnosing why message conversion, system-prompt injection, or tool formatting produced unexpected results. Large base64 image payloads are truncated to a `<base64 N chars>` placeholder so logs stay readable.

Combine with `LOG_FORMAT=json` for machine-parseable structured logs.

## Docker

Run Ollama and the proxy together with Docker Compose:

```bash
docker compose up -d
```

This starts:
- **ollama** — exposes port `11434`, persists models in a named volume
- **proxy** — exposes port `4000`, waits for Ollama to be healthy before starting

Override the model or add an API key without editing the compose file:

```bash
OLLAMA_MODEL=llama3.1:8b PROXY_API_KEY=mysecret docker compose up -d
```

Pull a model into the running Ollama container:

```bash
docker compose exec ollama ollama pull qwen2.5:7b
```

If you already have Ollama running on the host (not in Docker), build and run only the proxy:

```bash
docker build -t claude-ollama-proxy .
docker run -p 4000:4000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  claude-ollama-proxy
```

## systemd service (always-on, Linux)

Copy the repo, install the service, and start it:

```bash
sudo cp -r . /opt/claude-ollama-proxy
sudo cp claude-ollama-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now claude-ollama-proxy
```

To customise environment variables (model, port, API key) before enabling:

```bash
sudo systemctl edit claude-ollama-proxy
```

Add an `[Service]` override block, for example:

```ini
[Service]
Environment=OLLAMA_MODEL=qwen2.5-coder:7b
Environment=PROXY_API_KEY=mysecret
Environment=OLLAMA_NUM_CTX=32768
Environment=PROXY_WARMUP=true
```

Check status and logs:

```bash
sudo systemctl status claude-ollama-proxy
journalctl -u claude-ollama-proxy -f
```

## Hot reload (development)

Node 18+ supports `--watch` for automatic restarts on file change:

```bash
npm run dev
```

## Tests

```bash
npm test              # unit tests (test.js)
npm run test:integration  # integration tests against a real Ollama (test-integration.js)
npm run test:all       # both
```

## Per-request model selection

If you pass an Ollama model name (anything that doesn't start with `claude-`) as the `model` field in your request, the proxy will use it instead of `OLLAMA_MODEL`:

```bash
curl http://localhost:4000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: ollama' \
  -d '{
    "model": "qwen2.5-coder:7b",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Claude Code always sends a `claude-*` model name, so it will continue to use `OLLAMA_MODEL` (or its `MODEL_MAP` mapping) automatically.

## Token counting

```bash
curl http://localhost:4000/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-3-5-sonnet-20241022","messages":[{"role":"user","content":"Hello"}]}'
```

Returns:

```json
{ "input_tokens": 5 }
```

The proxy calls Ollama's `/api/tokenize` endpoint for accuracy and falls back to a character-based estimate (`chars / 4`) if the model isn't loaded yet. Claude Code uses this endpoint for context-window management.

## Context window

Ollama model defaults are often only 2048 tokens — far too small for a real Claude Code session. Always set `OLLAMA_NUM_CTX` to something appropriate for your hardware:

```bash
OLLAMA_NUM_CTX=32768 node proxy.js    # 32k — good for most tasks
OLLAMA_NUM_CTX=131072 node proxy.js   # 128k — for large codebases (needs more VRAM)
```

If long sessions still hit context-length errors, enable automatic history truncation:

```bash
OLLAMA_NUM_CTX=32768 PROXY_AUTO_TRUNCATE=true node proxy.js
```

When the estimated input would exceed `OLLAMA_NUM_CTX`, the proxy drops the oldest user/assistant turns (always preserving the system message and the most recent exchange) and sets an `x-context-truncated` response header with the number of messages dropped.

## Limitations

- Image blocks require a vision-capable model (e.g. `llava`, `qwen2.5-vl`); text-only models will error
- `thinking` block signatures are synthetic placeholders — they are not cryptographically signed by Ollama
- `think: true` forwarding requires Ollama 0.7+ and a model that supports native thinking (DeepSeek-R1, Qwen3-thinking)
- PDF and binary document blocks are converted to a placeholder note; only text-source documents are passed through
- The Messages Batch API is in-memory by default — set `PROXY_BATCH_PERSIST_PATH` to persist batches and results across restarts
