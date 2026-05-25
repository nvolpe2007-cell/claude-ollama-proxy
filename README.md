# claude-ollama-proxy

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
- Tool use / tool results (`tool_use`, `tool_result`) ↔ OpenAI function calling
- Image content blocks (base64 and URL sources) → OpenAI vision format
- `document` content blocks → plain text for Ollama
- Both streaming (SSE) and non-streaming responses
- Graceful errors when Ollama is offline

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
  Ollama: http://localhost:11434
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model to use for all requests |
| `PROXY_PORT` | `4000` | Port the proxy listens on |
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of your Ollama instance. Accepts a comma-separated list of URLs for round-robin load distribution across multiple Ollama instances/GPUs |
| `PROXY_API_KEY` | *(unset)* | If set, require this key on every API request |
| `MODEL_MAP` | *(unset)* | JSON map of `claude-*` names/prefixes to Ollama models (see below) |
| `PROXY_TLS_CERT` | *(unset)* | Path to PEM certificate file — enables HTTPS when set |
| `PROXY_TLS_KEY` | *(unset)* | Path to PEM private key file — required when cert is set |
| `CORS_ORIGIN` | `*` | Value for `Access-Control-Allow-Origin`; set to a specific origin to restrict browser access |
| `OLLAMA_NUM_CTX` | *(model default)* | Context window size sent to Ollama. Model defaults are often only 2048 — set to `32768` or higher for real sessions |
| `OLLAMA_KEEP_ALIVE` | *(Ollama default)* | How long the model stays loaded in GPU memory between requests (`"5m"`, `"0"` to unload immediately, `"-1"` to keep forever) |
| `PROXY_TIMEOUT` | *(none)* | Hard per-request timeout in milliseconds. If Ollama does not respond within this window the proxy aborts and returns a `504` error (non-streaming) or an SSE error event (streaming) |
| `PROXY_MAX_TOKENS` | `8192` | Default `max_tokens` applied when the client omits the field. Increase for models with larger output budgets |
| `PROXY_MAX_BODY_SIZE` | *(none)* | Hard limit on request body size in bytes. Requests that exceed this are rejected with `413`. Protects against runaway base64-image payloads. Example: `10485760` for 10 MB |
| `PROXY_SYSTEM_PROMPT` | *(unset)* | Operator-defined text prepended to every request's system prompt. When the client also supplies a system prompt, the proxy's text comes first, separated by two newlines. Useful for enforcing consistent model behavior without modifying client config |
| `PROXY_WARMUP` | `false` | When `true`, sends a minimal preflight request to Ollama after startup to pre-load the configured model into GPU memory, eliminating cold-start latency on the first real request |
| `RATE_LIMIT_RPM` | *(none)* | Global request rate limit in requests per minute across all callers. Applies to `POST /v1/messages` and `POST /v1/messages/count_tokens`. Returns `429` with `retry-after` header when exceeded |
| `RATE_LIMIT_PER_IP_RPM` | *(none)* | Per-client-IP rate limit in requests per minute. Uses `x-forwarded-for` when behind a reverse proxy. Both global and per-IP limits can be active simultaneously |
| `LOG_FORMAT` | `text` | Log format for request lines. `text` writes human-readable lines; `json` writes a single JSON object per request — useful for log aggregation tools like Grafana Loki, Datadog, or AWS CloudWatch |

Examples:

```bash
OLLAMA_MODEL=llama3.1:8b node proxy.js
OLLAMA_HOST=http://192.168.1.50:11434 node proxy.js   # remote Ollama instance
PROXY_API_KEY=mysecret node proxy.js                  # enable auth
OLLAMA_NUM_CTX=32768 PROXY_WARMUP=true node proxy.js  # production setup
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

The `/health` and `/metrics` endpoints are always unauthenticated so monitoring tools can reach them freely.

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

## List available models

```bash
curl http://localhost:4000/v1/models
```

Returns the models currently loaded in Ollama in OpenAI-compatible format:

```json
{
  "object": "list",
  "data": [
    { "id": "qwen2.5:7b",       "object": "model", "owned_by": "ollama" },
    { "id": "qwen2.5-coder:7b", "object": "model", "owned_by": "ollama" }
  ]
}
```

When `MODEL_MAP` is configured, Claude alias names (e.g. `claude-3-haiku`) are also included in the list so model-picker clients like Cursor and Continue can discover and select them.

Look up a specific model by ID:

```bash
curl http://localhost:4000/v1/models/qwen2.5:7b
```

## Check the proxy is running

```bash
curl http://localhost:4000/health
```

Returns `200 OK` when Ollama is reachable:

```json
{
  "status": "ok",
  "proxy": "running",
  "ollama": "reachable",
  "model": "qwen2.5:7b",
  "port": 4000,
  "timestamp": "2026-04-27T12:00:00.000Z"
}
```

Returns `503` with `"status": "degraded"` if Ollama is offline.

## Metrics

### JSON metrics

```bash
curl http://localhost:4000/metrics
```

Returns an in-memory snapshot including:
- Uptime, total request counts per route, HTTP status code breakdown
- p50/p95/p99 latency percentiles (rolling 1000-sample window), min/avg/max
- Cumulative input and output token totals
- Per-model breakdown of request counts and token usage
- Current active streaming connection count

### Prometheus metrics

```bash
curl http://localhost:4000/metrics/prometheus
```

Returns the same data in [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) (version 0.0.4), ready to be scraped directly by Prometheus without any additional exporter. Useful for Grafana dashboards.

Exposed metrics include `proxy_requests_total`, `proxy_http_responses_total`, `proxy_request_duration_ms`, `proxy_tokens_total`, `proxy_model_requests_total`, `proxy_model_tokens_total`, `proxy_active_streams`, and `proxy_errors_total`.

## OpenAI-compatible passthrough

The proxy also accepts native OpenAI chat format directly, making it a drop-in for OpenAI-compatible clients (Cursor, Continue, LiteLLM, etc.):

```bash
POST /v1/chat/completions
```

Requests are forwarded to Ollama with no format translation. All proxy features apply: authentication, rate limiting, retry, timeout, client-abort propagation, keepalive SSE comments, request logging, and per-model metrics. `MODEL_MAP` and `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` tuning are applied. Both streaming and non-streaming are supported.

## Rate limiting

Two independent rate limits can be set simultaneously. Both apply to `POST /v1/messages` and `POST /v1/messages/count_tokens`.

```bash
# Global cap: 60 requests/min across all callers
RATE_LIMIT_RPM=60 node proxy.js

# Per-IP cap: 10 requests/min per client
RATE_LIMIT_PER_IP_RPM=10 node proxy.js

# Both active at once
RATE_LIMIT_RPM=100 RATE_LIMIT_PER_IP_RPM=20 node proxy.js
```

When a limit is exceeded the proxy responds with `429 rate_limit_error` and sets `retry-after`, `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, and `x-ratelimit-reset-requests` headers — matching Anthropic's own API header naming.

## System prompt injection

Prepend a fixed system prompt to every request without modifying clients:

```bash
PROXY_SYSTEM_PROMPT="You are a helpful assistant. Always respond in English." node proxy.js
```

When the client already sends a system prompt, the proxy's text comes first, separated by two newlines. Handles string, array-of-blocks, and absent system prompts. Also applied to `POST /v1/messages/count_tokens` so token estimates reflect the injected text.

## Multi-host round-robin

Distribute load across multiple Ollama instances by setting `OLLAMA_HOST` to a comma-separated list:

```bash
OLLAMA_HOST=http://gpu1:11434,http://gpu2:11434,http://gpu3:11434 node proxy.js
```

Requests are distributed round-robin across all listed hosts. `GET /health` checks all hosts in parallel and reports per-host status in a `hosts` array.

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

## Thinking models

The proxy automatically extracts `<think>…</think>` blocks produced by reasoning models (DeepSeek-R1, Qwen3-thinking, etc.) and converts them to proper Anthropic `thinking` content blocks. Claude Code displays them as structured reasoning without any extra configuration.

When Claude Code enables extended thinking (sends `thinking: {type: "enabled", budget_tokens: N}`), the proxy forwards `think: true` to Ollama (Ollama 0.7+), which activates native thinking for supported models.

Thinking blocks in assistant messages are also preserved in multi-turn conversation history — they are re-encoded as `<think>…</think>` tags when the conversation history is sent back to Ollama, so reasoning context is maintained across turns with DeepSeek-R1, Qwen3-thinking, and similar models.

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

Claude Code always sends a `claude-*` model name, so it will continue to use `OLLAMA_MODEL` automatically.

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

## Limitations

- Image blocks require a vision-capable model (e.g. `llava`, `qwen2.5-vl`); text-only models will error
- `thinking` block signatures are synthetic placeholders — they are not cryptographically signed by Ollama
- `think: true` forwarding requires Ollama 0.7+ and a model that supports native thinking (DeepSeek-R1, Qwen3-thinking)
- PDF and binary document blocks are converted to a placeholder note; only text-source documents are passed through
