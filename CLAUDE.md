# claude-ollama-proxy — MTSM Nick

## What this is
A Node.js proxy that translates Anthropic API requests (Claude format) into Ollama/OpenAI format, so you can use Claude Code and other Anthropic tools against a local Ollama model.

## How it works
- Listens on port 4000 by default
- Accepts Anthropic `messages` API format (including tools/tool_use)
- Converts to OpenAI chat format that Ollama understands
- Forwards to Ollama at localhost:11434
- Translates responses back to Anthropic format

## Config
```
OLLAMA_MODEL=qwen2.5:7b        (default model)
OLLAMA_HOST=<url>[,<url>...]   (Ollama base URL; comma-separated list for round-robin multi-host; default http://localhost:11434)
PROXY_PORT=4000                (default port)
PROXY_API_KEY=<secret>         (optional; if set, enforces x-api-key / Bearer auth)
MODEL_MAP=<json>               (optional; maps claude-* names/prefixes to Ollama models)
PROXY_TLS_CERT=<path>          (optional; path to PEM cert file — enables HTTPS)
PROXY_TLS_KEY=<path>           (optional; path to PEM key file — required when cert is set)
CORS_ORIGIN=<origin>           (optional; Access-Control-Allow-Origin value; default '*')
OLLAMA_NUM_CTX=<n>             (optional; context window size sent to Ollama on every request; model default if unset — often only 2048, set to 32768+ for real sessions)
OLLAMA_KEEP_ALIVE=<duration>   (optional; how long Ollama holds the model in GPU memory between requests, e.g. "30m", "0" to unload immediately, "-1" to keep forever)
PROXY_TIMEOUT=<ms>             (optional; hard per-request timeout in milliseconds; proxy aborts and returns 504 / SSE error if Ollama takes longer; default is no timeout)
PROXY_MAX_TOKENS=<n>           (optional; default max_tokens when the client omits it; default 8192)
PROXY_MAX_BODY_SIZE=<bytes>    (optional; reject requests whose Content-Length exceeds this value with 413; default no limit; example: 10485760 for 10 MB)
LOG_FORMAT=<text|json>         (optional; 'text' emits human-readable lines (default); 'json' emits one JSON object per request for log aggregation tools — Grafana Loki, Datadog, CloudWatch, etc.)
```

### MODEL_MAP example
```
MODEL_MAP='{"claude-3-haiku":"qwen2.5:7b","claude-3-sonnet":"qwen2.5:14b","claude-3-opus":"qwen2.5:72b"}'
```
Exact match wins; if no exact match, any key that is a prefix of the requested model name is used (e.g. `"claude-3-haiku"` matches `claude-3-haiku-20240307`). Non-`claude-*` model names always pass through unchanged.

## How to run
```bash
node proxy.js
```
Then point Claude Code at http://localhost:4000 instead of the Anthropic API.

## What's implemented
- SSE streaming with full Anthropic event sequence
- tool_use / tool_result round-trip (streaming and non-streaming)
- Image content block support (base64 and URL sources → OpenAI vision format)
- Per-request model selection (pass any non-claude-* model name in the request)
- GET /v1/models — lists models available in Ollama
- GET /health — checks Ollama reachability, returns model + port
- Graceful error handling when Ollama is offline (502 with hint)
- Request logging (method, path, status, duration, tokens_in, tokens_out, model) to stdout; LOG_FORMAT=json emits machine-parseable JSON for log aggregation (Grafana Loki, Datadog, CloudWatch, etc.)
- Keepalive SSE comments every 15 s to survive reverse-proxy timeouts
- Graceful shutdown on SIGTERM / SIGINT
- Optional API key auth via PROXY_API_KEY (x-api-key or Authorization: Bearer)
- README.md with full setup instructions
- Dockerfile + docker-compose.yml (Ollama + proxy, health-check gated startup)
- claude-ollama-proxy.service — systemd unit for always-on Linux deployment
- Retry with exponential backoff on transient Ollama 5xx errors (up to 3 retries: 500 ms, 1 s, 2 s)
- Correct streaming output_tokens: message_delta deferred until trailing usage chunk is consumed
- top_k forwarding — passed through to Ollama's OpenAI-compat endpoint
- POST /v1/messages/count_tokens — uses Ollama /api/tokenize for accuracy, falls back to chars/4
- Streaming message_delta includes both input_tokens and output_tokens from trailing Ollama usage chunk
- Router-level try/catch prevents handler throws from becoming unhandled promise rejections (crash)
- Process-level uncaughtException/unhandledRejection handlers keep server alive on stray async errors
- URL routing strips query params (?foo=bar variants no longer 404)
- Non-streaming path guards against empty Ollama choices array
- MODEL_MAP env var — routes claude-* model names/prefixes to specific Ollama models; resolveModel() handles exact then prefix matching; startup log prints each mapping
- Client abort propagation — AbortController tied to client socket close; cancels in-flight Ollama fetch when caller disconnects (e.g. Ctrl+C in Claude Code), freeing GPU resources immediately; AbortErrors silently discarded; fetchWithRetry never retries them
- TLS / HTTPS support — set PROXY_TLS_CERT + PROXY_TLS_KEY to enable; proxy creates an https.createServer with those PEM files; startup logs TLS status; exits with a clear error if files cannot be read
- Streaming robustness: fallback for streams that end without an explicit finish_reason — defaults stop_reason to end_turn and closes any open content blocks before emitting message_delta/message_stop
- `<think>` tag extraction — both streaming and non-streaming paths detect `<think>…</think>` blocks from thinking models (DeepSeek-R1, Qwen3, etc.) and convert them to Anthropic `thinking` content blocks; streaming uses a state machine with tag-boundary buffering; thinking blocks carry a synthetic signature placeholder; zero-overhead fast path when no `<think>` tag is present
- Streaming thinking `signature_delta` — emits a `signature_delta` event before each `content_block_stop` for thinking blocks, completing the Anthropic extended-thinking streaming protocol so clients that validate thinking signatures don't reject the response
- Prompt-caching compat fields — all usage objects (streaming `message_start`, `message_delta`, and non-streaming response) include `cache_creation_input_tokens: 0` and `cache_read_input_tokens: 0`; Claude Code sends `anthropic-beta: prompt-caching-2024-07-31` on every request and expects these fields
- `document` content block support — Anthropic `document` blocks in user messages, system prompts, and tool results are converted to text for Ollama; text-source documents pass through directly; base64-encoded text documents are decoded; binary (PDF) and URL sources get an informative placeholder; block title is preserved as a header
- CORS support — all responses include Access-Control-Allow-Origin/Methods/Headers; OPTIONS preflight requests return 204 immediately so browser-based callers work without a separate CORS proxy; CORS_ORIGIN env var restricts the allowed origin (default '*')
- `seed` parameter forwarding — passed through to Ollama for reproducible outputs
- `disable_parallel_tool_use` forwarding — maps Anthropic's disable_parallel_tool_use:true to OpenAI's parallel_tool_calls:false
- `thinking` parameter forwarding — Anthropic's `thinking:{type:"enabled",budget_tokens:N}` maps to Ollama's `think:true` (Ollama 0.7+); supported models (Qwen3-thinking, DeepSeek-R1, etc.) natively emit `<think>` blocks which the proxy's existing state machine converts to Anthropic thinking content blocks
- `OLLAMA_NUM_CTX` env var — sets `num_ctx` on every Ollama request to override the model's default context window (often only 2048 tokens, far too small for Claude Code sessions); set to 32768+ in production
- `OLLAMA_KEEP_ALIVE` env var — sets `keep_alive` on every Ollama request to control how long the model stays loaded in GPU memory between requests; useful for tuning GPU utilisation vs. latency
- `OLLAMA_HOST` env var — overrides the Ollama base URL (default `http://localhost:11434`); accepts a comma-separated list of URLs for round-robin load distribution across multiple Ollama instances/GPUs; `getOllamaHost()` picks the next host in rotation; each request handler captures its host at the start so retries within a single request always hit the same host; GET /health checks all configured hosts in parallel and reports per-host status in a `hosts` array while keeping backward-compat `ollama`/`ollamaError` fields derived from the first host
- Thinking block round-trip in conversation history — `thinking` content blocks in assistant messages are converted back to `<think>…</think>` tags when sending conversation history to Ollama, so multi-turn sessions with thinking models (DeepSeek-R1, Qwen3-thinking) preserve full chain-of-thought context across turns
- GET /metrics — in-memory request metrics endpoint: uptime, per-route request counts, HTTP status code breakdown, p50/p95/p99 latency percentiles (rolling 1000-sample window), cumulative input/output token totals, and current active streaming connection count; no auth required (operational data only)
- `PROXY_TIMEOUT` env var — optional hard per-request timeout (ms); if Ollama does not complete within this window the proxy aborts the in-flight fetch and returns a 504 JSON error (non-streaming) or an SSE error event (streaming); timeout fires via AbortController reusing the existing client-abort signal so GPU resources are released immediately; timedOut flag distinguishes timeout from client disconnect so the response path can send the correct error instead of silently closing; default is no timeout
- `PROXY_MAX_TOKENS` env var — configurable default max_tokens applied when the client omits the field (default 8192); useful for models with larger output budgets or strict token limits
- GET /v1/models/:modelId — looks up a single model by ID from Ollama's model list; returns the same object shape as GET /v1/models entries; 404 if the model isn't in Ollama; 502 if Ollama is unreachable; colon-separated names (e.g. `qwen2.5:7b`) are URL-decoded automatically
- Input validation — POST /v1/messages now validates that `messages` is present and is an array, returning a 400 `invalid_request_error` (with a descriptive message) instead of crashing into a 500; `stream` field now correctly defaults to `false` per the Anthropic API spec when not specified by the client
- `request-id` response header — every response carries a unique `req_`-prefixed identifier that matches Anthropic's API header naming; useful for correlating proxy logs with client-side errors
- `PROXY_MAX_BODY_SIZE` env var — optional hard limit on request body size (bytes); if the client's Content-Length header exceeds the limit the proxy immediately returns 413 `request_too_large` without reading the body, protecting against runaway base64-image payloads; default is no limit
