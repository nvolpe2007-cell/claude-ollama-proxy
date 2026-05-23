const http  = require('http');
const https = require('https');
const fs    = require('fs');

// Multi-host: OLLAMA_HOST may be a comma-separated list of Ollama base URLs.
// Requests are distributed round-robin across all listed hosts so you can
// spread load across multiple GPUs or Ollama instances.
const OLLAMA_HOSTS = (process.env.OLLAMA_HOST || 'http://localhost:11434')
  .split(',').map(h => h.trim()).filter(Boolean);

// Round-robin index. Node.js is single-threaded so no lock is needed.
let _hostIdx = 0;
function getOllamaHost() {
  const host = OLLAMA_HOSTS[_hostIdx];
  _hostIdx = (_hostIdx + 1) % OLLAMA_HOSTS.length;
  return host;
}

const MODEL         = process.env.OLLAMA_MODEL    || 'qwen2.5:7b';
const PORT          = process.env.PROXY_PORT      || 4000;
const PROXY_API_KEY = process.env.PROXY_API_KEY   || null;
const TLS_CERT      = process.env.PROXY_TLS_CERT  || null;
const TLS_KEY       = process.env.PROXY_TLS_KEY   || null;
// CORS_ORIGIN controls the Access-Control-Allow-Origin header.
// Set to a specific origin (e.g. "https://my-app.example.com") or leave as "*" for open access.
const CORS_ORIGIN   = process.env.CORS_ORIGIN     || '*';
// LOG_FORMAT controls request log output. 'text' (default) emits a human-readable line;
// 'json' emits a single-line JSON object per request, useful for log aggregation tools
// (Grafana Loki, Datadog, CloudWatch, etc.).
const LOG_FORMAT    = process.env.LOG_FORMAT       || 'text';
// Ollama-specific tuning defaults applied to every request.
// num_ctx controls the context window — Ollama model defaults (often 2048) are too small
// for real Claude Code sessions; set this to at least 32768 in production.
const OLLAMA_NUM_CTX    = process.env.OLLAMA_NUM_CTX    ? Number(process.env.OLLAMA_NUM_CTX)    : null;
// keep_alive controls how long Ollama holds the model in GPU memory between requests.
// Use "0" to unload immediately, "-1" to keep forever, "30m" for 30 minutes, etc.
const OLLAMA_KEEP_ALIVE = process.env.OLLAMA_KEEP_ALIVE || null;
// Hard per-request timeout (ms). If set, the proxy aborts the Ollama request and returns
// a 504 / SSE error after this many milliseconds. Unset by default (no timeout).
const PROXY_TIMEOUT     = process.env.PROXY_TIMEOUT     ? Number(process.env.PROXY_TIMEOUT)     : null;
// Default max_tokens when the client does not specify one. 8192 is a safe default for most
// Ollama models; set higher (e.g. 32768) for models with large output budgets.
const PROXY_MAX_TOKENS  = process.env.PROXY_MAX_TOKENS  ? Number(process.env.PROXY_MAX_TOKENS)  : 8192;
// Optional system prompt injected before every request's system field.
// Useful for enforcing consistent model behavior across all callers without modifying clients.
// When the client also supplies a system prompt, the proxy's prompt is prepended (separated by
// two newlines). For array-form system prompts the proxy text becomes the first content block.
const PROXY_SYSTEM_PROMPT = process.env.PROXY_SYSTEM_PROMPT || null;
// Optional hard body-size limit (bytes). Requests exceeding this via Content-Length are
// rejected with 413 before the body is read, protecting against runaway base64-image payloads.
// Default is no limit. Example: PROXY_MAX_BODY_SIZE=10485760 for 10 MB.
const PROXY_WARMUP       = process.env.PROXY_WARMUP === 'true';
const PROXY_MAX_BODY_SIZE = (() => {
  if (!process.env.PROXY_MAX_BODY_SIZE) return null;
  const n = Number(process.env.PROXY_MAX_BODY_SIZE);
  if (!Number.isFinite(n) || n <= 0) {
    console.warn('Warning: PROXY_MAX_BODY_SIZE is not a valid positive number, ignoring');
    return null;
  }
  return n;
})();

// Optional request rate limits. Both apply only to POST /v1/messages and
// POST /v1/messages/count_tokens. Unset (disabled) by default.
// RATE_LIMIT_RPM        — global cap across all callers (requests / minute).
// RATE_LIMIT_PER_IP_RPM — per-client-IP cap (requests / minute); uses
//                         x-forwarded-for when behind a reverse proxy.
const RATE_LIMIT_RPM        = process.env.RATE_LIMIT_RPM        ? Number(process.env.RATE_LIMIT_RPM)        : null;
const RATE_LIMIT_PER_IP_RPM = process.env.RATE_LIMIT_PER_IP_RPM ? Number(process.env.RATE_LIMIT_PER_IP_RPM) : null;

// Optional JSON map: claude-* model name (or prefix) → Ollama model name.
// Exact match wins; then prefix match (e.g. "claude-3-haiku" matches any claude-3-haiku-*).
// Non-claude-* names in requests always pass through as-is regardless of this map.
// Example: MODEL_MAP='{"claude-3-haiku":"qwen2.5:7b","claude-3-opus":"qwen2.5:72b"}'
let MODEL_MAP = {};
try {
  if (process.env.MODEL_MAP) MODEL_MAP = JSON.parse(process.env.MODEL_MAP);
} catch (e) {
  console.warn('Warning: MODEL_MAP is not valid JSON, ignoring:', e.message);
}

// ── In-memory request metrics ────────────────────────────────────────────────
const _metrics = {
  startTime:    Date.now(),
  requests:     {},   // 'METHOD /path' → count
  statusCodes:  {},   // '200' → count
  latencies:    [],   // last 1000 durations (ms) for percentile calculations
  tokensIn:     0,
  tokensOut:    0,
  activeStreams: 0,
  errors:       0,
  modelsUsed:   {},   // 'model-name' → { requests, tokensIn, tokensOut }
};

function recordRequest(method, path, status, ms) {
  const k = `${method} ${path}`;
  _metrics.requests[k] = (_metrics.requests[k] || 0) + 1;
  const s = String(status);
  _metrics.statusCodes[s] = (_metrics.statusCodes[s] || 0) + 1;
  if (status >= 500) _metrics.errors++;
  if (_metrics.latencies.length >= 1000) _metrics.latencies.shift();
  _metrics.latencies.push(ms);
}

function recordTokens(input, output, model) {
  _metrics.tokensIn  += input;
  _metrics.tokensOut += output;
  if (model) {
    if (!_metrics.modelsUsed[model])
      _metrics.modelsUsed[model] = { requests: 0, tokensIn: 0, tokensOut: 0 };
    _metrics.modelsUsed[model].requests  += 1;
    _metrics.modelsUsed[model].tokensIn  += input;
    _metrics.modelsUsed[model].tokensOut += output;
  }
}

// Emits one log line per completed request. Meta carries optional token counts and model name
// populated by handleMessages; other routes leave it null.
// fmt defaults to the module-level LOG_FORMAT constant; tests may pass 'text'/'json' explicitly.
function logRequest(req, res, path, ms, meta, fmt = LOG_FORMAT) {
  if (fmt === 'json') {
    const entry = {
      ts:         new Date().toISOString(),
      method:     req.method,
      path,
      status:     res.statusCode,
      ms,
      request_id: res.getHeader('request-id') || undefined,
    };
    if (meta?.model)             entry.model      = meta.model;
    if (meta?.tokensIn  != null) entry.tokens_in  = meta.tokensIn;
    if (meta?.tokensOut != null) entry.tokens_out = meta.tokensOut;
    console.log(JSON.stringify(entry));
  } else {
    const toks = meta?.tokensIn != null
      ? ` in=${meta.tokensIn} out=${meta.tokensOut}` + (meta.model ? ` model=${meta.model}` : '')
      : '';
    console.log(`${req.method} ${req.url} ${res.statusCode} ${ms}ms${toks}`);
  }
}

function pctile(sorted, p) {
  if (!sorted.length) return 0;
  return sorted[Math.ceil(p / 100 * sorted.length) - 1];
}

function resolveModel(requestedModel) {
  if (!requestedModel) return MODEL;
  if (MODEL_MAP[requestedModel]) return MODEL_MAP[requestedModel];
  if (requestedModel.startsWith('claude-')) {
    for (const [key, target] of Object.entries(MODEL_MAP)) {
      if (requestedModel.startsWith(key)) return target;
    }
    return MODEL;
  }
  return requestedModel;
}

// Merges PROXY_SYSTEM_PROMPT with the request's system field.
// Handles string, array-of-blocks, and absent system prompts.
// When PROXY_SYSTEM_PROMPT is unset, returns the original system value unchanged.
function injectSystemPrompt(system) {
  if (!PROXY_SYSTEM_PROMPT) return system;
  if (!system) return PROXY_SYSTEM_PROMPT;
  if (typeof system === 'string') return `${PROXY_SYSTEM_PROMPT}\n\n${system}`;
  // Array of Anthropic content blocks — prepend a text block.
  return [{ type: 'text', text: PROXY_SYSTEM_PROMPT }, ...system];
}

// Anthropic messages/tools → OpenAI format
function toOpenAIMessages(messages, system) {
  const result = [];

  const systemText = Array.isArray(system)
    ? system.map(b => b.type === 'document' ? documentBlockToText(b) : (b.text || '')).filter(Boolean).join('\n')
    : system;
  if (systemText) result.push({ role: 'system', content: systemText });

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      result.push({ role: msg.role, content: msg.content });
      continue;
    }

    const blocks = Array.isArray(msg.content) ? msg.content : [];
    const toolResults = blocks.filter(b => b.type === 'tool_result');
    const toolUses = blocks.filter(b => b.type === 'tool_use');
    const textParts = blocks.filter(b => b.type === 'text');

    if (toolResults.length > 0) {
      // tool result messages become role:tool in OpenAI
      for (const tr of toolResults) {
        if (!tr.tool_use_id) continue;
        const rawContent = Array.isArray(tr.content)
          ? tr.content.map(c => {
              if (c.type === 'document') return documentBlockToText(c) || '';
              return c.text || '';
            }).join('')
          : (tr.content || '');
        const content = tr.is_error ? `[ERROR] ${rawContent}` : rawContent;
        result.push({ role: 'tool', tool_call_id: tr.tool_use_id, content });
      }
      if (textParts.length > 0) {
        result.push({ role: msg.role, content: textParts.map(b => b.text).join('') });
      }
    } else {
      const rawText    = textParts.map(b => b.text).join('');
      const imageParts = blocks.filter(b => b.type === 'image');
      const docParts   = blocks.filter(b => b.type === 'document');
      const docTexts   = docParts.map(documentBlockToText).filter(Boolean);
      // Re-wrap thinking blocks as <think> tags so Ollama sees prior reasoning in context.
      // Without this, multi-turn conversations with thinking models (DeepSeek-R1, Qwen3-thinking)
      // lose all previous chain-of-thought on every follow-up request.
      const thinkParts = blocks.filter(b => b.type === 'thinking');
      const thinkText  = thinkParts.map(b => `<think>${b.thinking || ''}</think>`).join('');
      const bodyText   = [rawText, ...docTexts].filter(Boolean).join('\n\n');
      const text       = [thinkText, bodyText].filter(Boolean).join('\n');

      let content;
      if (imageParts.length > 0) {
        content = [];
        if (text) content.push({ type: 'text', text });
        for (const img of imageParts) {
          const converted = imageBlockToOpenAI(img);
          if (converted) content.push(converted);
        }
      } else {
        content = text || '';
      }

      const out = { role: msg.role, content };
      if (toolUses.length > 0) {
        out.tool_calls = toolUses.map(tu => ({
          id: tu.id,
          type: 'function',
          function: { name: tu.name, arguments: JSON.stringify(tu.input) }
        }));
      }
      result.push(out);
    }
  }

  return result;
}

function imageBlockToOpenAI(block) {
  const src = block.source;
  if (!src) return null;
  if (src.type === 'base64') {
    return { type: 'image_url', image_url: { url: `data:${src.media_type};base64,${src.data}` } };
  }
  if (src.type === 'url') {
    return { type: 'image_url', image_url: { url: src.url } };
  }
  return null;
}

// Converts an Anthropic `document` content block to a plain-text string for Ollama.
// Text/plain documents are decoded from base64 if needed; PDFs and other binary types
// get a placeholder note since Ollama has no native PDF parser.
function documentBlockToText(block) {
  const src = block.source;
  if (!src) return null;
  const header = block.title ? `[Document: ${block.title}]\n` : '';
  if (src.type === 'text') {
    return header + (src.data || '');
  }
  if (src.type === 'base64') {
    if (src.media_type && src.media_type.startsWith('text/')) {
      try { return header + Buffer.from(src.data, 'base64').toString('utf8'); } catch { return null; }
    }
    return header + `[Binary document (${src.media_type}) — not supported by this proxy]`;
  }
  if (src.type === 'url') {
    return header + `[Document URL: ${src.url} — not fetched by this proxy]`;
  }
  return null;
}

function toOpenAITools(tools) {
  if (!tools || tools.length === 0) return undefined;
  return tools.map(t => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }));
}

// Anthropic tool_choice → OpenAI tool_choice
function toOpenAIToolChoice(tc) {
  if (!tc) return undefined;
  switch (tc.type) {
    case 'auto': return 'auto';
    case 'none': return 'none';
    // "any" = must call at least one tool → OpenAI "required"
    case 'any':  return 'required';
    // "tool" = force a specific function
    case 'tool': return { type: 'function', function: { name: tc.name } };
    default:     return undefined;
  }
}

function sendSSE(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function newMsgId() {
  return 'msg_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// Splits raw model text into alternating thinking/text parts by <think>…</think> tags.
// Returns null when no <think> tag is present (fast path for non-thinking models).
function extractThinkingParts(text) {
  if (!text.includes('<think>')) return null;
  const parts = [];
  const re = /<think>([\s\S]*?)<\/think>/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    const before = text.slice(last, m.index).trim();
    if (before) parts.push({ type: 'text', text: before });
    parts.push({ type: 'thinking', thinking: m[1] });
    last = re.lastIndex;
  }
  const after = text.slice(last).trim();
  if (after) parts.push({ type: 'text', text: after });
  return parts.length ? parts : null;
}

// Sets CORS headers on every response so browser-based callers work without a proxy.
// Prefers a specific origin when CORS_ORIGIN is set; defaults to wildcard.
function setCORSHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', CORS_ORIGIN);
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers',
    'Content-Type, x-api-key, Authorization, anthropic-version, anthropic-beta');
  res.setHeader('Access-Control-Max-Age', '86400');
}

// Returns true if the request is authorised (or auth is disabled).
// Writes a 401 and returns false if the key is wrong.
function checkAuth(req, res) {
  if (!PROXY_API_KEY) return true;
  const fromHeader = req.headers['x-api-key']
    || (req.headers['authorization'] || '').replace(/^Bearer\s+/i, '').trim();
  if (fromHeader === PROXY_API_KEY) return true;
  res.writeHead(401, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: { type: 'authentication_error', message: 'Invalid or missing API key' } }));
  return false;
}

async function readBody(req) {
  if (PROXY_MAX_BODY_SIZE) {
    const cl = Number(req.headers['content-length'] || 0);
    if (cl > PROXY_MAX_BODY_SIZE) {
      const err = new Error(`Request body ${cl} B exceeds limit of ${PROXY_MAX_BODY_SIZE} B (PROXY_MAX_BODY_SIZE)`);
      err.code = 'PAYLOAD_TOO_LARGE';
      throw err;
    }
  }
  let body = '';
  let bytesRead = 0;
  for await (const chunk of req) {
    if (PROXY_MAX_BODY_SIZE) {
      bytesRead += Buffer.isBuffer(chunk) ? chunk.length : Buffer.byteLength(chunk);
      if (bytesRead > PROXY_MAX_BODY_SIZE) {
        const err = new Error(`Request body exceeds limit of ${PROXY_MAX_BODY_SIZE} B (PROXY_MAX_BODY_SIZE)`);
        err.code = 'PAYLOAD_TOO_LARGE';
        throw err;
      }
    }
    body += chunk;
  }
  return body;
}

// ── Rate limiting ────────────────────────────────────────────────────────────
// Fixed-window counters keyed by 'global' or a client IP string.
// Each window is exactly 60 seconds wide. A new window starts on first request
// after the previous one expires, which is cheap and good enough for burst protection.
const _rateLimitWindows = new Map(); // key → { count, windowStart }

// Returns the client IP, respecting x-forwarded-for for reverse-proxy deployments.
function getClientIp(req) {
  const xff = req.headers['x-forwarded-for'];
  if (xff) return xff.split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}

// Checks the given key against the limit. Attaches x-ratelimit-* headers to res.
// Returns true if the request is within limits; writes a 429 and returns false if not.
function checkRateLimit(key, limit, req, res) {
  const now = Date.now();
  let w = _rateLimitWindows.get(key);
  if (!w || now - w.windowStart >= 60_000) {
    w = { count: 0, windowStart: now };
    _rateLimitWindows.set(key, w);
  }
  w.count++;
  const remaining  = Math.max(0, limit - w.count);
  const resetEpoch = Math.ceil((w.windowStart + 60_000) / 1000);
  res.setHeader('x-ratelimit-limit-requests',     String(limit));
  res.setHeader('x-ratelimit-remaining-requests', String(remaining));
  res.setHeader('x-ratelimit-reset-requests',     String(resetEpoch));
  if (w.count > limit) {
    const retryAfter = Math.max(1, Math.ceil((w.windowStart + 60_000 - now) / 1000));
    res.setHeader('retry-after', String(retryAfter));
    if (!res.headersSent) res.writeHead(429, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      type: 'error',
      error: {
        type: 'rate_limit_error',
        message: `Rate limit exceeded (${limit} req/min). Retry after ${retryAfter}s.`,
      },
    }));
    return false;
  }
  return true;
}

// Retry fetch on transient Ollama 5xx errors or network failures.
// Does NOT retry 4xx (client errors), AbortErrors, or once the streaming body has begun.
async function fetchWithRetry(url, options, maxRetries = 3, baseDelay = 500) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, options);
      if (res.status < 500 || attempt >= maxRetries) return res;
      console.warn(`Ollama ${res.status}, retrying (${attempt + 1}/${maxRetries})…`);
    } catch (e) {
      if (e.name === 'AbortError') throw e;  // never retry client-initiated aborts
      if (attempt >= maxRetries) throw e;
      console.warn(`Ollama fetch error: ${e.message}, retrying (${attempt + 1}/${maxRetries})…`);
    }
    // Jitter (±25%) prevents thundering herd when multiple retries fire concurrently.
    const jitter = 0.75 + Math.random() * 0.5;
    await new Promise(r => setTimeout(r, baseDelay * 2 ** attempt * jitter));
  }
}

async function handleMessages(req, res) {
  const ollamaBase = getOllamaHost();

  // Abort Ollama request if the client disconnects — avoids wasting GPU compute.
  const ac = new AbortController();

  // Optional hard timeout: abort Ollama fetch and surface an error when PROXY_TIMEOUT is set.
  // Uses a separate flag so the response path can distinguish timeout from client-disconnect.
  let timedOut = false;
  let _timeoutId = null;
  const clearTO = () => { if (_timeoutId) { clearTimeout(_timeoutId); _timeoutId = null; } };
  if (PROXY_TIMEOUT) {
    _timeoutId = setTimeout(() => {
      timedOut = true;
      if (!res.writableEnded) ac.abort();
      console.warn(`Request timeout after ${PROXY_TIMEOUT}ms — aborting Ollama request`);
    }, PROXY_TIMEOUT);
  }

  const onClientClose = () => { if (!res.writableEnded) { clearTO(); ac.abort(); } };
  req.socket.once('close', onClientClose);

  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (!Array.isArray(anthropicReq.messages)) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`messages` is required and must be an array' } }));
    return;
  }

  // Anthropic API spec: stream defaults to false when not specified.
  const streaming = anthropicReq.stream === true;

  // Use request model if it looks like an Ollama model name (not a claude-* alias).
  // This lets callers switch models per-request without restarting the proxy.
  const effectiveModel = resolveModel(anthropicReq.model);

  const openaiReq = {
    model: effectiveModel,
    messages: toOpenAIMessages(anthropicReq.messages, injectSystemPrompt(anthropicReq.system)),
    stream: streaming,
    max_tokens: anthropicReq.max_tokens || PROXY_MAX_TOKENS,
    ...(streaming && { stream_options: { include_usage: true } }),
  };

  const tools = toOpenAITools(anthropicReq.tools);
  if (tools) openaiReq.tools = tools;
  const toolChoice = toOpenAIToolChoice(anthropicReq.tool_choice);
  if (toolChoice !== undefined) openaiReq.tool_choice = toolChoice;
  if (anthropicReq.temperature !== undefined) openaiReq.temperature = anthropicReq.temperature;
  if (anthropicReq.top_p     !== undefined) openaiReq.top_p     = anthropicReq.top_p;
  // top_k is an Anthropic parameter; pass through to Ollama's OpenAI-compat layer
  if (anthropicReq.top_k     !== undefined) openaiReq.top_k     = anthropicReq.top_k;
  if (anthropicReq.seed      !== undefined) openaiReq.seed      = anthropicReq.seed;
  if (anthropicReq.stop_sequences?.length) openaiReq.stop = anthropicReq.stop_sequences;
  // Anthropic's disable_parallel_tool_use maps to OpenAI's parallel_tool_calls: false
  if (anthropicReq.disable_parallel_tool_use === true) openaiReq.parallel_tool_calls = false;
  // Anthropic extended-thinking → Ollama's native think parameter.
  // Ollama 0.7+ passes think:true to supported models (Qwen3-thinking, DeepSeek-R1, etc.)
  // which makes them emit <think>…</think> blocks the proxy already handles.
  if (anthropicReq.thinking?.type === 'enabled') openaiReq.think = true;
  // Apply global Ollama tuning defaults (overrideable per-deployment via env vars).
  if (OLLAMA_NUM_CTX)    openaiReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) openaiReq.keep_alive = OLLAMA_KEEP_ALIVE;

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(openaiReq),
      signal: ac.signal,
      // No hard timeout on the request itself — models can be slow.
      // Connection refused / DNS failures are caught below.
    });
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    if (e.name === 'AbortError') {
      if (timedOut) {
        if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
        if (!res.writableEnded) res.end(JSON.stringify({
          error: { type: 'request_timeout', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` }
        }));
      } else {
        res.end();
      }
      return;
    }
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    const hint = isConnRefused
      ? ' — is Ollama running? Try: ollama serve'
      : '';
    res.writeHead(502);
    res.end(JSON.stringify({ error: { type: 'ollama_unreachable', message: e.message + hint } }));
    return;
  }

  if (!ollamaRes.ok) {
    const err = await ollamaRes.text();
    res.writeHead(502);
    res.end(JSON.stringify({ error: err }));
    return;
  }

  // ── Non-streaming ───────────────────────────────────────────────────────────────────────────
  if (!streaming) {
    let data;
    try {
      data = await ollamaRes.json();
    } catch (e) {
      req.socket.off('close', onClientClose);
      clearTO();
      if (e.name === 'AbortError') {
        if (timedOut) {
          if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
          if (!res.writableEnded) res.end(JSON.stringify({
            error: { type: 'request_timeout', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` }
          }));
        } else {
          res.end();
        }
        return;
      }
      throw e;
    }
    req.socket.off('close', onClientClose);
    clearTO();
    const choice = data.choices?.[0];
    if (!choice) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'ollama_error', message: 'Empty choices in Ollama response' } }));
      return;
    }
    const msg = choice.message;

    const content = [];
    if (msg.content) {
      const thinkParts = extractThinkingParts(msg.content);
      if (thinkParts) {
        for (const p of thinkParts) {
          if (p.type === 'thinking') {
            // signature is an opaque field required by the Anthropic spec for round-trips;
            // we use a placeholder since Ollama has no cryptographic thinking signing.
            content.push({ type: 'thinking', thinking: p.thinking, signature: 'ollama-proxy-extracted' });
          } else {
            content.push({ type: 'text', text: p.text });
          }
        }
      } else {
        content.push({ type: 'text', text: msg.content });
      }
    }
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        let input = {};
        try { input = JSON.parse(tc.function.arguments); } catch {}
        content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
      }
    }

    const promptTok = data.usage?.prompt_tokens || 0;
    const completionTok = data.usage?.completion_tokens || 0;
    recordTokens(promptTok, completionTok, effectiveModel);
    res._logMeta = { model: effectiveModel, tokensIn: promptTok, tokensOut: completionTok };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      id: newMsgId(),
      type: 'message',
      role: 'assistant',
      content,
      model: effectiveModel,
      stop_reason: choice.finish_reason === 'tool_calls' ? 'tool_use'
               : choice.finish_reason === 'length'     ? 'max_tokens'
               : 'end_turn',
      stop_sequence: null,
      usage: {
        input_tokens: data.usage?.prompt_tokens || 0,
        output_tokens: data.usage?.completion_tokens || 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0
      }
    }));
    return;
  }

  // ── Streaming ───────────────────────────────────────────────────────────────────────────────
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });
  _metrics.activeStreams++;

  const id = newMsgId();
  sendSSE(res, 'message_start', {
    type: 'message_start',
    message: {
      id, type: 'message', role: 'assistant', content: [],
      model: effectiveModel, stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }
    }
  });

  let textBlockOpen = false;
  let textIndex = 0;           // anthropic block index of the text block (captured on first open)
  const toolBlocks = {};       // openai tool index → { anthropicIndex, id, name, args }
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = null;       // set on finish_reason; message_delta deferred until after loop

  // State machine for routing <think>…</think> content to Anthropic thinking blocks.
  // Models like DeepSeek-R1 and Qwen3-thinking prefix their response with a <think> block.
  let thinkState = 'initial';  // 'initial' | 'thinking' | 'text'
  let thinkBuf   = '';         // chars pending routing (straddle tag boundaries)
  let thinkCount = 0;          // thinking blocks fully closed so far

  function routeThinkChunk(flush) {
    for (;;) {
      if (!thinkBuf.length) break;

      if (thinkState === 'initial') {
        const tag = '<think>';
        if (thinkBuf.startsWith(tag)) {
          sendSSE(res, 'content_block_start', {
            type: 'content_block_start', index: thinkCount,
            content_block: { type: 'thinking', thinking: '' }
          });
          sendSSE(res, 'ping', { type: 'ping' });
          thinkBuf = thinkBuf.slice(tag.length);
          thinkState = 'thinking';
          continue;
        }
        // Partial prefix of '<think>' — wait for more data unless flushing.
        if (!flush && tag.startsWith(thinkBuf)) break;
        // Not a think tag — treat remainder as plain text.
        thinkState = 'text';
        continue;
      }

      if (thinkState === 'thinking') {
        const etag = '</think>';
        const ei = thinkBuf.indexOf(etag);
        if (ei !== -1) {
          if (ei > 0) sendSSE(res, 'content_block_delta', {
            type: 'content_block_delta', index: thinkCount,
            delta: { type: 'thinking_delta', thinking: thinkBuf.slice(0, ei) }
          });
          sendSSE(res, 'content_block_delta', {
            type: 'content_block_delta', index: thinkCount,
            delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
          });
          sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkCount });
          thinkCount++;
          thinkBuf = thinkBuf.slice(ei + etag.length);
          thinkState = 'initial';
          continue;
        }
        // No closing tag yet. Hold back enough chars to detect a split '</think>'.
        const lt = thinkBuf.lastIndexOf('<');
        const safe = flush
          ? thinkBuf.length
          : (lt > 0 ? lt : Math.max(0, thinkBuf.length - etag.length));
        if (safe > 0) {
          sendSSE(res, 'content_block_delta', {
            type: 'content_block_delta', index: thinkCount,
            delta: { type: 'thinking_delta', thinking: thinkBuf.slice(0, safe) }
          });
          thinkBuf = thinkBuf.slice(safe);
        }
        break;
      }

      // thinkState === 'text'
      if (!textBlockOpen) {
        textIndex = thinkCount;  // thinking blocks, if any, occupy lower indices
        sendSSE(res, 'content_block_start', {
          type: 'content_block_start', index: textIndex,
          content_block: { type: 'text', text: '' }
        });
        sendSSE(res, 'ping', { type: 'ping' });
        textBlockOpen = true;
      }
      sendSSE(res, 'content_block_delta', {
        type: 'content_block_delta', index: textIndex,
        delta: { type: 'text_delta', text: thinkBuf }
      });
      thinkBuf = '';
      break;
    }
  }

  // Prevent reverse-proxy read timeouts on slow models.
  const keepAlive = setInterval(() => res.writableEnded || res.write(': keepalive\n\n'), 15_000);

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') continue;

        let chunk;
        try { chunk = JSON.parse(raw); } catch { continue; }

        if (chunk.usage) {
          inputTokens  = chunk.usage.prompt_tokens     || inputTokens;
          outputTokens = chunk.usage.completion_tokens || outputTokens;
        }

        const choice = chunk.choices?.[0];
        if (!choice) continue;
        const delta = choice.delta;

        // Text delta — route through <think> state machine
        if (delta.content) {
          thinkBuf += delta.content;
          routeThinkChunk(false);
        }

        // Tool call deltas
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const oi = tc.index; // openai index
            if (!toolBlocks[oi]) {
              const ai = thinkCount + (textBlockOpen ? 1 : 0) + oi;
              toolBlocks[oi] = {
                anthropicIndex: ai,
                id: tc.id || `toolu_${oi}`,
                name: tc.function?.name || '',
                args: ''
              };
              sendSSE(res, 'content_block_start', {
                type: 'content_block_start', index: ai,
                content_block: {
                  type: 'tool_use',
                  id: toolBlocks[oi].id,
                  name: toolBlocks[oi].name,
                  input: {}
                }
              });
              sendSSE(res, 'ping', { type: 'ping' });
            }

            const tb = toolBlocks[oi];
            if (tc.function?.name && !tb.name) tb.name = tc.function.name;
            if (tc.id && tb.id.startsWith('toolu_')) tb.id = tc.id;

            if (tc.function?.arguments) {
              tb.args += tc.function.arguments;
              sendSSE(res, 'content_block_delta', {
                type: 'content_block_delta', index: tb.anthropicIndex,
                delta: { type: 'input_json_delta', partial_json: tc.function.arguments }
              });
            }
          }
        }

        // Finish — close content blocks now but defer terminal events until after
        // the loop so the trailing usage chunk (sent by Ollama after finish_reason)
        // is processed first, giving us the correct outputTokens value.
        if (choice.finish_reason) {
          stopReason = choice.finish_reason === 'tool_calls' ? 'tool_use'
                     : choice.finish_reason === 'length'     ? 'max_tokens'
                     : 'end_turn';

          routeThinkChunk(true);
          if (thinkState === 'thinking') {
            sendSSE(res, 'content_block_delta', {
              type: 'content_block_delta', index: thinkCount,
              delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
            });
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkCount });
            thinkCount++;
          }
          if (textBlockOpen) {
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textIndex });
          }
          for (const tb of Object.values(toolBlocks)) {
            sendSSE(res, 'content_block_stop', {
              type: 'content_block_stop', index: tb.anthropicIndex
            });
          }
        }
      }
    }
    // All chunks consumed — emit terminal events with correct token counts.
    // Guard against streams that end without an explicit finish_reason.
    if (!stopReason) {
      stopReason = 'end_turn';
      routeThinkChunk(true);
      if (thinkState === 'thinking') {
        sendSSE(res, 'content_block_delta', {
          type: 'content_block_delta', index: thinkCount,
          delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
        });
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkCount });
        thinkCount++;
      }
      if (textBlockOpen) {
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textIndex });
      }
      for (const tb of Object.values(toolBlocks)) {
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: tb.anthropicIndex });
      }
    }
    if (!res.writableEnded) {
      sendSSE(res, 'message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { input_tokens: inputTokens, output_tokens: outputTokens, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }
      });
      sendSSE(res, 'message_stop', { type: 'message_stop' });
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      if (timedOut && !res.writableEnded) {
        sendSSE(res, 'error', {
          type: 'error',
          error: { type: 'request_timeout', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` }
        });
      }
      // Client-disconnect AbortErrors are silently discarded.
    } else {
      console.error('Stream error:', e.message);
      if (!res.writableEnded) {
        sendSSE(res, 'error', { type: 'error', error: { type: 'stream_error', message: e.message } });
      }
    }
  } finally {
    req.socket.off('close', onClientClose);
    clearTO();
    clearInterval(keepAlive);
    _metrics.activeStreams--;
  }

  recordTokens(inputTokens, outputTokens, effectiveModel);
  res._logMeta = { model: effectiveModel, tokensIn: inputTokens, tokensOut: outputTokens };
  res.end();
}

async function handleModels(req, res) {
  const ollamaBase = getOllamaHost();
  let data;
  try {
    const r = await fetch(`${ollamaBase}/api/tags`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`Ollama returned HTTP ${r.status}`);
    data = await r.json();
  } catch (e) {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_unreachable', message: e.message } }));
    return;
  }

  const models = (data.models || []).map(m => ({
    id: m.name,
    object: 'model',
    created: m.modified_at ? Math.floor(new Date(m.modified_at).getTime() / 1000) : 0,
    owned_by: 'ollama'
  }));

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ object: 'list', data: models }));
}

async function handleModelById(req, res, modelId) {
  const ollamaBase = getOllamaHost();
  let data;
  try {
    const r = await fetch(`${ollamaBase}/api/tags`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error(`Ollama returned HTTP ${r.status}`);
    data = await r.json();
  } catch (e) {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_unreachable', message: e.message } }));
    return;
  }

  const model = (data.models || []).find(m => m.name === modelId);
  if (!model) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Model '${modelId}' not found in Ollama` } }));
    return;
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    id: model.name,
    object: 'model',
    created: model.modified_at ? Math.floor(new Date(model.modified_at).getTime() / 1000) : 0,
    owned_by: 'ollama'
  }));
}

async function handleCountTokens(req, res) {
  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch { res.writeHead(400); res.end('{"error":"bad json"}'); return; }

  const effectiveModel = resolveModel(anthropicReq.model);
  const ollamaBase = getOllamaHost();

  // Flatten messages + system to a single string for tokenization.
  // Tool schemas are appended as JSON since they consume context.
  const messages = toOpenAIMessages(anthropicReq.messages || [], injectSystemPrompt(anthropicReq.system));
  let prompt = messages.map(m => {
    if (typeof m.content === 'string') return m.content;
    if (Array.isArray(m.content)) return m.content.map(p => p.text || '').join('');
    return '';
  }).join('\n');
  if (anthropicReq.tools?.length) prompt += '\n' + JSON.stringify(anthropicReq.tools);

  // Try Ollama's native tokenize endpoint; fall back to char/4 estimate.
  let inputTokens;
  try {
    const r = await fetch(`${ollamaBase}/api/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: effectiveModel, prompt }),
      signal: AbortSignal.timeout(5000)
    });
    inputTokens = r.ok
      ? (await r.json()).tokens?.length ?? Math.ceil(prompt.length / 4)
      : Math.ceil(prompt.length / 4);
  } catch {
    inputTokens = Math.ceil(prompt.length / 4);
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ input_tokens: inputTokens }));
}

async function handleHealth(req, res) {
  const hostResults = await Promise.all(OLLAMA_HOSTS.map(async url => {
    try {
      const check = await fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(3000) });
      return { url, status: check.ok ? 'ok' : 'unreachable', error: check.ok ? undefined : `HTTP ${check.status}` };
    } catch (e) {
      return { url, status: 'unreachable', error: e.message };
    }
  }));
  const anyOk = hostResults.some(h => h.status === 'ok');
  const allOk = hostResults.every(h => h.status === 'ok');
  // Backward-compat fields derived from the first host (single-host deployments unchanged).
  const first = hostResults[0];

  res.writeHead(anyOk ? 200 : 503, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    status: allOk ? 'ok' : 'degraded',
    proxy: 'running',
    hosts: hostResults,
    ollama: first.status === 'ok' ? 'reachable' : 'unreachable',
    ollamaError: first.error || undefined,
    model: MODEL,
    port: Number(PORT),
    timestamp: new Date().toISOString()
  }));
}

async function handleMetrics(req, res) {
  const sorted = [..._metrics.latencies].sort((a, b) => a - b);
  const latSum = sorted.reduce((a, b) => a + b, 0);
  const latAvg = sorted.length ? Math.round((latSum / sorted.length) * 100) / 100 : 0;
  const modelsUsage = {};
  for (const [model, m] of Object.entries(_metrics.modelsUsed)) {
    modelsUsage[model] = { requests: m.requests, tokens_in: m.tokensIn, tokens_out: m.tokensOut };
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    uptime_seconds:      Math.floor((Date.now() - _metrics.startTime) / 1000),
    requests_total:      _metrics.requests,
    status_codes:        _metrics.statusCodes,
    latency_p50_ms:      pctile(sorted, 50),
    latency_p95_ms:      pctile(sorted, 95),
    latency_p99_ms:      pctile(sorted, 99),
    latency_min_ms:      sorted.length ? sorted[0] : 0,
    latency_max_ms:      sorted.length ? sorted[sorted.length - 1] : 0,
    latency_avg_ms:      latAvg,
    tokens_input_total:  _metrics.tokensIn,
    tokens_output_total: _metrics.tokensOut,
    active_streams:      _metrics.activeStreams,
    errors_total:        _metrics.errors,
    models_usage:        modelsUsage,
  }, null, 2));
}

// Prometheus text exposition format (https://prometheus.io/docs/instrumenting/exposition_formats/)
// Scraped by Prometheus at GET /metrics/prometheus; compatible with Grafana Loki.
async function handleMetricsPrometheus(req, res) {
  const sorted = [..._metrics.latencies].sort((a, b) => a - b);
  const uptime  = Math.floor((Date.now() - _metrics.startTime) / 1000);
  const latSum  = _metrics.latencies.reduce((a, b) => a + b, 0);

  // Escape label values per the Prometheus exposition format spec.
  function lv(s) { return String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n'); }

  const out = [];

  out.push('# HELP proxy_uptime_seconds Seconds since proxy process started');
  out.push('# TYPE proxy_uptime_seconds gauge');
  out.push(`proxy_uptime_seconds ${uptime}`);
  out.push('');

  out.push('# HELP proxy_requests_total Total requests handled, partitioned by HTTP method and route path');
  out.push('# TYPE proxy_requests_total counter');
  for (const [key, count] of Object.entries(_metrics.requests)) {
    const sp     = key.indexOf(' ');
    const method = key.slice(0, sp);
    const path   = key.slice(sp + 1);
    out.push(`proxy_requests_total{method="${lv(method)}",path="${lv(path)}"} ${count}`);
  }
  out.push('');

  out.push('# HELP proxy_http_responses_total Total HTTP responses, partitioned by status code');
  out.push('# TYPE proxy_http_responses_total counter');
  for (const [code, count] of Object.entries(_metrics.statusCodes)) {
    out.push(`proxy_http_responses_total{code="${lv(code)}"} ${count}`);
  }
  out.push('');

  out.push('# HELP proxy_request_duration_ms Request latency summary over rolling 1000-sample window');
  out.push('# TYPE proxy_request_duration_ms summary');
  out.push(`proxy_request_duration_ms{quantile="0.5"} ${pctile(sorted, 50)}`);
  out.push(`proxy_request_duration_ms{quantile="0.95"} ${pctile(sorted, 95)}`);
  out.push(`proxy_request_duration_ms{quantile="0.99"} ${pctile(sorted, 99)}`);
  out.push(`proxy_request_duration_ms_sum ${latSum}`);
  out.push(`proxy_request_duration_ms_count ${_metrics.latencies.length}`);
  out.push('');

  out.push('# HELP proxy_tokens_total Cumulative LLM tokens, partitioned by direction');
  out.push('# TYPE proxy_tokens_total counter');
  out.push(`proxy_tokens_total{direction="input"} ${_metrics.tokensIn}`);
  out.push(`proxy_tokens_total{direction="output"} ${_metrics.tokensOut}`);
  out.push('');

  out.push('# HELP proxy_active_streams Current number of open SSE streaming connections');
  out.push('# TYPE proxy_active_streams gauge');
  out.push(`proxy_active_streams ${_metrics.activeStreams}`);
  out.push('');

  out.push('# HELP proxy_model_requests_total Total completed LLM requests per model');
  out.push('# TYPE proxy_model_requests_total counter');
  for (const [model, m] of Object.entries(_metrics.modelsUsed)) {
    out.push(`proxy_model_requests_total{model="${lv(model)}"} ${m.requests}`);
  }
  out.push('');

  out.push('# HELP proxy_model_tokens_total Cumulative LLM tokens per model partitioned by direction');
  out.push('# TYPE proxy_model_tokens_total counter');
  for (const [model, m] of Object.entries(_metrics.modelsUsed)) {
    out.push(`proxy_model_tokens_total{model="${lv(model)}",direction="input"} ${m.tokensIn}`);
    out.push(`proxy_model_tokens_total{model="${lv(model)}",direction="output"} ${m.tokensOut}`);
  }
  out.push('');

  out.push('# HELP proxy_errors_total Total 5xx responses (server-side errors)');
  out.push('# TYPE proxy_errors_total counter');
  out.push(`proxy_errors_total ${_metrics.errors}`);
  out.push('');

  const latAvg = sorted.length ? Math.round((latSum / sorted.length) * 100) / 100 : 0;
  out.push('# HELP proxy_request_latency_min_ms Minimum observed request latency over rolling sample window');
  out.push('# TYPE proxy_request_latency_min_ms gauge');
  out.push(`proxy_request_latency_min_ms ${sorted.length ? sorted[0] : 0}`);
  out.push('');

  out.push('# HELP proxy_request_latency_max_ms Maximum observed request latency over rolling sample window');
  out.push('# TYPE proxy_request_latency_max_ms gauge');
  out.push(`proxy_request_latency_max_ms ${sorted.length ? sorted[sorted.length - 1] : 0}`);
  out.push('');

  out.push('# HELP proxy_request_latency_avg_ms Mean request latency over rolling sample window');
  out.push('# TYPE proxy_request_latency_avg_ms gauge');
  out.push(`proxy_request_latency_avg_ms ${latAvg}`);
  out.push('');

  res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4; charset=utf-8' });
  res.end(out.join('\n') + '\n');
}

async function requestHandler(req, res) {
  // CORS headers on every response — must happen before any writeHead call.
  setCORSHeaders(res);
  // Unique per-request ID surfaces in logs and lets callers correlate proxy-side errors.
  res.setHeader('request-id', `req_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`);

  // Respond to browser preflight checks immediately, before auth or body parsing.
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const start = Date.now();
  // Strip query string before routing so ?foo=bar variants still match.
  const path = req.url.split('?')[0];
  // handleMessages sets this before res ends so the finish handler can log token counts.
  res._logMeta = null;
  res.on('finish', () => {
    const ms = Date.now() - start;
    logRequest(req, res, path, ms, res._logMeta);
    recordRequest(req.method, path, res.statusCode, ms);
  });

  try {
    if (req.method === 'POST' && path === '/v1/messages') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      await handleMessages(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      await handleCountTokens(req, res);
    } else if (req.method === 'GET' && path === '/v1/models') {
      if (!checkAuth(req, res)) return;
      await handleModels(req, res);
    } else if (req.method === 'GET' && path.startsWith('/v1/models/')) {
      if (!checkAuth(req, res)) return;
      await handleModelById(req, res, decodeURIComponent(path.slice('/v1/models/'.length)));
    } else if (req.method === 'GET' && path === '/health') {
      await handleHealth(req, res);
    } else if (req.method === 'GET' && path === '/metrics') {
      await handleMetrics(req, res);
    } else if (req.method === 'GET' && path === '/metrics/prometheus') {
      await handleMetricsPrometheus(req, res);
    } else {
      res.writeHead(404);
      res.end('{"error":"not found"}');
    }
  } catch (e) {
    console.error('Unhandled request error:', e);
    if (!res.headersSent) {
      if (e.code === 'PAYLOAD_TOO_LARGE') {
        res.writeHead(413, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { type: 'request_too_large', message: e.message } }));
      } else {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { type: 'internal_error', message: e.message } }));
      }
    } else if (!res.writableEnded) {
      res.end();
    }
  }
}

if (require.main === module) {
  let server;
  if (TLS_CERT && TLS_KEY) {
    let tlsOpts;
    try {
      tlsOpts = { cert: fs.readFileSync(TLS_CERT), key: fs.readFileSync(TLS_KEY) };
    } catch (e) {
      console.error(`Fatal: cannot read TLS cert/key: ${e.message}`);
      process.exit(1);
    }
    server = https.createServer(tlsOpts, requestHandler);
  } else {
    server = http.createServer(requestHandler);
  }

  // Keep the process alive if a stray async error escapes a request handler.
  // Node 15+ crashes on unhandledRejection by default; log and continue instead.
  process.on('uncaughtException', (e) => console.error('Uncaught exception:', e));
  process.on('unhandledRejection', (reason) => console.error('Unhandled rejection:', reason));

  server.listen(PORT, () => {
    console.log(`\n  Claude-Ollama proxy ready`);
    console.log(`  Model : ${MODEL}`);
    if (Object.keys(MODEL_MAP).length > 0) {
      for (const [k, v] of Object.entries(MODEL_MAP))
        console.log(`  Map   : ${k} → ${v}`);
    }
    console.log(`  Port  : ${PORT}`);
    if (OLLAMA_HOSTS.length === 1) {
      console.log(`  Ollama: ${OLLAMA_HOSTS[0]}`);
    } else {
      console.log(`  Ollama: round-robin across ${OLLAMA_HOSTS.length} hosts:`);
      for (const h of OLLAMA_HOSTS) console.log(`    - ${h}`);
    }
    console.log(`  Auth  : ${PROXY_API_KEY ? 'enabled (PROXY_API_KEY set)' : 'disabled (open access)'}`);
    console.log(`  TLS   : ${TLS_CERT ? `enabled (cert: ${TLS_CERT})` : 'disabled (HTTP)'}`);
    console.log(`  CORS  : Access-Control-Allow-Origin: ${CORS_ORIGIN}`);
    console.log(`  Ctx   : ${OLLAMA_NUM_CTX ? `num_ctx=${OLLAMA_NUM_CTX}` : 'model default (set OLLAMA_NUM_CTX to override)'}`);
    if (OLLAMA_KEEP_ALIVE) console.log(`  Keep  : keep_alive=${OLLAMA_KEEP_ALIVE}`);
    console.log(`  Timeout: ${PROXY_TIMEOUT ? `${PROXY_TIMEOUT}ms per request` : 'none (set PROXY_TIMEOUT to limit)'}`);
    console.log(`  MaxTok: default max_tokens=${PROXY_MAX_TOKENS} (set PROXY_MAX_TOKENS to change)`);
    console.log(`  MaxBody: ${PROXY_MAX_BODY_SIZE ? `${PROXY_MAX_BODY_SIZE} B per request` : 'unlimited (set PROXY_MAX_BODY_SIZE to limit)'}`);
    if (PROXY_SYSTEM_PROMPT) console.log(`  SysPrompt: ${PROXY_SYSTEM_PROMPT.slice(0, 80)}${PROXY_SYSTEM_PROMPT.length > 80 ? '…' : ''}`);
    console.log(`  Logs  : format=${LOG_FORMAT} (set LOG_FORMAT=json for structured logging)`);
    console.log(`  Warmup: ${PROXY_WARMUP ? 'enabled — pre-loading model on startup' : 'disabled (set PROXY_WARMUP=true to pre-load model)'}`);
    const rlGlobal = RATE_LIMIT_RPM        ? `global ${RATE_LIMIT_RPM} req/min`    : 'no global limit';
    const rlIp     = RATE_LIMIT_PER_IP_RPM ? `per-IP ${RATE_LIMIT_PER_IP_RPM} req/min` : 'no per-IP limit';
    console.log(`  RateLimit: ${rlGlobal}; ${rlIp} (set RATE_LIMIT_RPM / RATE_LIMIT_PER_IP_RPM)\n`);

    if (PROXY_WARMUP) {
      // Fire warmup asynchronously so the server is already accepting connections while we load.
      // Uses setImmediate to yield back to the event loop first.
      setImmediate(async () => {
        const ollamaBase = OLLAMA_HOSTS[0];
        console.log(`  Warmup: sending preflight to ${ollamaBase} to load ${MODEL} into GPU memory…`);
        try {
          const r = await fetch(`${ollamaBase}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: MODEL,
              messages: [{ role: 'user', content: '.' }],
              max_tokens: 1,
              stream: false,
              ...(OLLAMA_KEEP_ALIVE && { keep_alive: OLLAMA_KEEP_ALIVE }),
            }),
            // Allow up to 3 minutes — large models can take time on first load.
            signal: AbortSignal.timeout(180_000),
          });
          if (r.ok) {
            console.log('  Warmup: model ready — GPU memory loaded\n');
          } else {
            const text = await r.text().catch(() => '');
            console.warn(`  Warmup: Ollama returned HTTP ${r.status} — first real request will retry: ${text.slice(0, 120)}\n`);
          }
        } catch (e) {
          console.warn(`  Warmup: failed (${e.message}) — first request will load model on demand\n`);
        }
      });
    }
  });

  // closeIdleConnections drops idle keep-alive connections immediately so
  // server.close() can actually reach its callback on SIGTERM.  Without this,
  // any open keep-alive socket from Claude Code would prevent a clean exit.
  function shutdown() {
    server.closeIdleConnections();
    server.close(() => process.exit(0));
    setTimeout(() => server.closeAllConnections(), 10_000).unref();
  }
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

module.exports = {
  resolveModel,
  toOpenAIMessages,
  toOpenAITools,
  toOpenAIToolChoice,
  extractThinkingParts,
  documentBlockToText,
  imageBlockToOpenAI,
  injectSystemPrompt,
  logRequest,
  getOllamaHost,
  OLLAMA_HOSTS,
  requestHandler,
  handleMetricsPrometheus,
  // Rate-limit internals exported for unit testing only.
  checkRateLimit,
  getClientIp,
  _rateLimitWindows,
};
