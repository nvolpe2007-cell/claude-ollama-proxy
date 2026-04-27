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

## What to work on next
1. Improve tool_use / tool_result round-trip support
2. Add streaming response support
3. Add model selection endpoint
4. Write a README with setup instructions
5. Add error handling for Ollama being offline
6. Push to GitHub so it's accessible from other machines
