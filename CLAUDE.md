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
OLLAMA_MODEL=qwen2.5:7b   (default model)
PROXY_PORT=4000            (default port)
```

## How to run
```bash
node proxy.js
```
Then point Claude Code at http://localhost:4000 instead of the Anthropic API.

## What's implemented
- SSE streaming with full Anthropic event sequence
- tool_use / tool_result round-trip (streaming and non-streaming)
- Per-request model selection (pass any non-claude-* model name in the request)
- GET /v1/models — lists models available in Ollama
- GET /health — checks Ollama reachability, returns model + port
- Graceful error handling when Ollama is offline (502 with hint)
- Request logging (method, path, status, duration) to stdout
- Keepalive SSE comments every 15 s to survive reverse-proxy timeouts
- Graceful shutdown on SIGTERM / SIGINT
- README.md with full setup instructions

## What to work on next
- Image content block support (convert Anthropic image blocks to OpenAI vision format)
- Optional request auth (validate x-api-key so the proxy isn't fully open)
- Docker / systemd service file for always-on deployment
