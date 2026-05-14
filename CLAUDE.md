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
PROXY_PORT=4000                (default port)
PROXY_API_KEY=<secret>         (optional; if set, enforces x-api-key / Bearer auth)
MODEL_MAP=<json>               (optional; maps claude-* names/prefixes to Ollama models)
PROXY_TLS_CERT=<path>          (optional; path to PEM cert file — enables HTTPS)
PROXY_TLS_KEY=<path>           (optional; path to PEM key file — required when cert is set)
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
- Request logging (method, path, status, duration) to stdout
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
