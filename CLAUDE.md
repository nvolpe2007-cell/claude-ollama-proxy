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
```

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

## What to work on next
- TLS / HTTPS support (or document Caddy / nginx reverse-proxy setup)
