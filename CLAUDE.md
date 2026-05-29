# claude-ollama-proxy — MTSM Nick

## What this is
A Node.js proxy that translates Anthropic API requests (Claude format) into Ollama/OpenAI format, so you can use Claude Code and other Anthropic tools against a local Ollama model. Also accepts OpenAI-format requests directly, making it a drop-in proxy for OpenAI-compatible clients (Cursor, Continue, LiteLLM, etc.).

## How it works
- Listens on port 4000 by default
- Accepts Anthropic `messages` API format (including tools/tool_use) — `POST /v1/messages`
- Also accepts OpenAI chat format directly — `POST /v1/chat/completions` (no translation, piped through)
- Also accepts legacy OpenAI text completions format — `POST /v1/completions` (converts `prompt` → single user message, returns text_completion envelope; supports streaming)
- Accepts OpenAI embeddings format — `POST /v1/embeddings` (forwarded to Ollama's `/api/embed`)
- Converts Anthropic requests to OpenAI chat format that Ollama understands
- Forwards to Ollama at localhost:11434
- Translates responses back to Anthropic format (OpenAI-format requests are piped verbatim)

## Config

All config is read from environment variables. The proxy also automatically loads a `.env` file from the current working directory (or the script's own directory) at startup — variables already set in the shell environment always take precedence (12-factor style).

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
PROXY_MAX_BODY_SIZE=<bytes>    (optional; reject requests whose body exceeds this value with 413; Content-Length header is checked immediately (before reading the body), and actual bytes are counted during streaming so clients that omit Content-Length are also enforced; default no limit; example: 10485760 for 10 MB)
LOG_FORMAT=<text|json>         (optional; 'text' emits human-readable lines (default); 'json' emits one JSON object per request for log aggregation tools — Grafana Loki, Datadog, CloudWatch, etc.)
LOG_LEVEL=<info|debug>         (optional; 'info' logs one summary line per request (default); 'debug' also logs the full translated OpenAI-format request body sent to Ollama and, for non-streaming calls, the raw Ollama response body — invaluable for diagnosing why message conversion, system-prompt injection, or tool formatting produces unexpected results; large base64 image payloads are automatically truncated to a `<base64 N chars>` placeholder so logs stay readable)
PROXY_WARMUP=true              (optional; when 'true', sends a minimal preflight request to Ollama after startup to pre-load the model into GPU memory, eliminating cold-start latency on the first real request; default false)
RATE_LIMIT_RPM=<n>            (optional; global request rate limit in requests/minute across all callers; applies to POST /v1/messages and POST /v1/messages/count_tokens; returns 429 with retry-after header; default no limit)
RATE_LIMIT_PER_IP_RPM=<n>     (optional; per-client-IP rate limit in requests/minute; uses x-forwarded-for when behind a reverse proxy; both limits can be active simultaneously; default no limit)
PROXY_SYSTEM_PROMPT=<text>    (optional; text prepended to every request's system prompt; if the client already sends a system prompt the proxy's text comes first, separated by two newlines; useful for enforcing consistent model behavior across all callers without modifying client config)
OLLAMA_OPTIONS=<json>         (optional; JSON object of arbitrary Ollama model parameters applied to every request — repeat_penalty, mirostat, num_gpu, tfs_z, typical_p, etc.; per-request client params take precedence; OLLAMA_NUM_CTX and OLLAMA_KEEP_ALIVE take highest precedence over this; example: {"repeat_penalty":1.1,"mirostat":2,"num_gpu":33})
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
- `.env` file loading — proxy reads a `.env` file from `process.cwd()` (or `__dirname` if different) at startup; shell environment variables always take precedence; supports `KEY=VALUE`, single/double-quoted values, `#` comments, and blank lines; `parseDotEnv()` helper exported for unit testing
- SSE streaming with full Anthropic event sequence
- tool_use / tool_result round-trip (streaming and non-streaming)
- Image content block support (base64 and URL sources → OpenAI vision format)
- Per-request model selection (pass any non-claude-* model name in the request)
- GET /v1/models — lists models available in Ollama; when MODEL_MAP is configured, also exposes the mapped Claude alias names (e.g. `claude-3-haiku`) so model-picker clients (Cursor, Continue, OpenWebUI) can discover and select them without knowing the underlying Ollama model name; aliases that clash with a real Ollama model ID are suppressed to avoid duplicates
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
- GET /v1/models/:modelId — looks up a single model by ID from Ollama's model list; also resolves MODEL_MAP aliases (exact and prefix) so `GET /v1/models/claude-3-haiku` returns the aliased model when MODEL_MAP is configured — now consistent with GET /v1/models which already exposed aliases; 404 if the model isn't in Ollama or the alias target isn't; 502 if Ollama is unreachable; colon-separated names (e.g. `qwen2.5:7b`) are URL-decoded automatically
- `version` field in GET /health — proxy package.json version included in every health response and the GET / live dashboard Config card for easier debugging and monitoring
- `OLLAMA_KEEP_ALIVE` exposed in dashboard — the configured keep_alive value now shows in the Config card of the GET / live dashboard alongside num_ctx and other tuning knobs
- Input validation — POST /v1/messages now validates that `messages` is present and is an array, returning a 400 `invalid_request_error` (with a descriptive message) instead of crashing into a 500; `stream` field now correctly defaults to `false` per the Anthropic API spec when not specified by the client
- `request-id` response header — every response carries a unique `req_`-prefixed identifier that matches Anthropic's API header naming; useful for correlating proxy logs with client-side errors
- `PROXY_MAX_BODY_SIZE` env var — optional hard limit on request body size (bytes); enforced at two points: Content-Length header is checked immediately before reading (fast rejection for well-behaved clients), and actual bytes are counted during body streaming so clients that omit Content-Length are also enforced; returns 413 `request_too_large`; protects against runaway base64-image payloads; default is no limit
- GET /metrics/prometheus — Prometheus text exposition format (version 0.0.4) of the same metrics as GET /metrics; exposes `proxy_uptime_seconds`, `proxy_requests_total{method,path}`, `proxy_http_responses_total{code}`, `proxy_request_duration_ms` summary (p50/p95/p99 + _sum + _count), `proxy_tokens_total{direction}`, `proxy_active_streams`, `proxy_errors_total`, and `proxy_request_latency_{min,max,avg}_ms` gauges matching the JSON /metrics fields; scraped directly by Prometheus without any exporter; Content-Type `text/plain; version=0.0.4; charset=utf-8`
- `PROXY_WARMUP` env var — when set to `true`, fires a minimal preflight request to Ollama immediately after the server starts listening; this pre-loads the configured model into GPU memory so the first real Claude Code request incurs no cold-start latency; uses a 3-minute timeout (large models can take time on first load); warmup failures are logged as warnings and do not prevent the proxy from serving traffic
- Retry jitter — exponential backoff delays now include ±25% random jitter to prevent thundering herd when multiple concurrent requests all retry at the same moment (especially relevant in multi-host round-robin deployments)
- Extended latency metrics — GET /metrics now includes `latency_min_ms`, `latency_max_ms`, and `latency_avg_ms` alongside the existing p50/p95/p99 percentiles for a more complete picture of request latency distribution
- Per-model usage metrics — GET /metrics includes `models_usage` object breaking down request counts, input tokens, and output tokens by Ollama model name; GET /metrics/prometheus exposes `proxy_model_requests_total{model}` and `proxy_model_tokens_total{model,direction}` counters; especially useful with MODEL_MAP deployments to compare usage across models
- Rate limiting — `RATE_LIMIT_RPM` (global) and `RATE_LIMIT_PER_IP_RPM` (per caller IP) fixed-window request caps applied to POST /v1/messages and POST /v1/messages/count_tokens; exceeded limit returns 429 `rate_limit_error` with `retry-after` header; every rate-limited response also carries `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, and `x-ratelimit-reset-requests` headers matching Anthropic's own API header naming; per-IP extraction respects `x-forwarded-for` for reverse-proxy deployments; both limits may be active simultaneously; disabled by default
- `PROXY_SYSTEM_PROMPT` env var — optional operator-defined system prompt prepended to every request; merged before the client's own system field so the client's instructions still take effect; handles string, array-of-blocks, and absent system prompts; propagated to POST /v1/messages, POST /v1/messages/count_tokens, and POST /v1/chat/completions (OpenAI passthrough) so all callers — Claude Code, Cursor, Continue, LiteLLM — get consistent model behavior; token estimates reflect the injected text; startup log prints a truncated preview when set
- `tool_result` image content support — `image` blocks inside `tool_result` messages (e.g., screenshots from computer-use tools) are extracted and appended as a follow-up `user` message with OpenAI multipart image_url format, since OpenAI `role:tool` messages only support string content; text and image follow-up parts are merged into a single user message; vision-capable Ollama models (LLaVA, Qwen2-VL, etc.) can therefore see tool-returned images in multi-turn sessions
- `POST /v1/chat/completions` OpenAI-format passthrough — accepts native OpenAI chat format and pipes it directly to Ollama with no translation; applies full proxy infrastructure (auth, rate-limiting, retry, timeout, client-abort, keepalive SSE comments, request logging, and per-model metrics); `stream:true` is supported and SSE is piped verbatim line-by-line; MODEL_MAP and `OLLAMA_NUM_CTX`/`OLLAMA_KEEP_ALIVE` tuning are applied; makes the proxy a drop-in for any OpenAI-compatible client (Cursor, Continue, LiteLLM, etc.) alongside its existing Anthropic-format support
- `POST /v1/embeddings` OpenAI-format embeddings — forwards to Ollama's `/api/embed` endpoint; accepts `input` as a string or array of strings; returns OpenAI-compatible `{ object:"list", data:[{object:"embedding", embedding:[...], index}], model, usage }` envelope; applies auth, rate-limiting, timeout, retry, client-abort, CORS, and request-id headers same as other endpoints; supports all Ollama embedding models (nomic-embed-text, mxbai-embed-large, etc.) and resolves MODEL_MAP aliases
- `GET /` live dashboard — self-contained HTML status page served by the proxy itself (zero external dependencies); auto-refreshes every 5 seconds; shows Ollama connectivity, active streaming connections, startup config (model, port, auth, TLS, rate limits, context window, etc.), cumulative token usage, per-route request counts, HTTP status code breakdown, p50/p95/p99/min/max/avg latency, and per-model usage; links to `/health`, `/metrics`, `/metrics/prometheus`, and `/v1/models`; also handles `GET /favicon.ico` with 204 to suppress browser console noise
- `OLLAMA_OPTIONS` env var — optional JSON object of arbitrary Ollama model parameters (e.g. `repeat_penalty`, `mirostat`, `mirostat_eta`, `mirostat_tau`, `num_gpu`, `num_thread`, `tfs_z`, `typical_p`) applied to every outbound request; per-request client values take precedence; `OLLAMA_NUM_CTX` and `OLLAMA_KEEP_ALIVE` still take highest precedence; applied to POST /v1/messages, POST /v1/chat/completions, and POST /v1/embeddings; validated at startup with a clear warning if the value is not a JSON object; startup log and live dashboard show the active options; `parseOllamaOptions()` helper exported for unit testing
- `LOG_LEVEL=debug` verbose mode — when set to `debug`, logs the full translated OpenAI-format request body sent to Ollama (via `[DEBUG] → Ollama` lines) and, for non-streaming requests, the raw Ollama response body (`[DEBUG] ← Ollama response`); large base64 image payloads are automatically replaced with a `<base64 N chars>` placeholder via `sanitizeForLog()` so logs stay readable even with vision requests; startup log and dashboard Config card both show the active log level; complementary to `LOG_FORMAT` which controls log-line structure; `sanitizeForLog()` helper exported for unit testing
- `POST /v1/completions` legacy text completions — accepts the OpenAI text completions format (`prompt` string or array, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `stream`); converts `prompt` into a single user message and forwards to Ollama's `/v1/chat/completions`; converts the response back to the `text_completion` envelope (`object:"text_completion"`, `choices[].text`); streaming emits SSE deltas in completions format; applies full proxy infrastructure (auth, rate-limiting, retry, timeout, client-abort, keepalive, CORS, metrics, MODEL_MAP, OLLAMA_NUM_CTX/KEEP_ALIVE, PROXY_SYSTEM_PROMPT, OLLAMA_OPTIONS); makes the proxy compatible with older tools and LiteLLM configs that target `/v1/completions`
- Multi-host warmup fix — `PROXY_WARMUP=true` now warms all hosts in `OLLAMA_HOST` in parallel (previously only `OLLAMA_HOSTS[0]` was warmed, leaving all other GPUs in a multi-host deployment cold until their first real request)
