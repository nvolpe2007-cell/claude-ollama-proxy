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

Examples:

```bash
OLLAMA_MODEL=llama3.1:8b node proxy.js
OLLAMA_HOST=http://192.168.1.50:11434 node proxy.js   # remote Ollama instance
```

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
- No authentication — intended for local use only
