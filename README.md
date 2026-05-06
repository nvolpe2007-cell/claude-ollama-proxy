# claude-ollama-proxy

A lightweight Node.js proxy that translates Anthropic API requests (Claude format) into Ollama's OpenAI-compatible format. This lets you point **Claude Code** — or any tool that speaks the Anthropic `messages` API — at a local Ollama model instead of the real Claude API.

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
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of your Ollama instance |
| `PROXY_API_KEY` | *(unset)* | If set, require this key on every API request |

Examples:

```bash
OLLAMA_MODEL=llama3.1:8b node proxy.js
OLLAMA_HOST=http://192.168.1.50:11434 node proxy.js   # remote Ollama instance
PROXY_API_KEY=mysecret node proxy.js                  # enable auth
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

The `/health` endpoint is always unauthenticated so monitoring tools can reach it freely.

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

## Recommended models

Smaller models that work well with Claude Code's tool-use patterns:

| Model | Pull command | Notes |
|---|---|---|
| `qwen2.5:7b` | `ollama pull qwen2.5:7b` | Default; good balance of speed and quality |
| `qwen2.5-coder:7b` | `ollama pull qwen2.5-coder:7b` | Better for coding tasks |
| `llama3.1:8b` | `ollama pull llama3.1:8b` | Good general purpose |
| `mistral:7b` | `ollama pull mistral:7b` | Fast, lower memory |

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

## Limitations

- `top_k` is not forwarded (Ollama accepts it via `options`, not the OpenAI-compat layer)
- Image blocks require a vision-capable model (e.g. `llava`, `qwen2.5-vl`); text-only models will error
- No TLS — use a reverse proxy (nginx, Caddy) if exposing beyond localhost
