const http  = require('http');
const https = require('https');
const fs    = require('fs');
const path  = require('path');

let PROXY_VERSION = 'unknown';
try { PROXY_VERSION = require('./package.json').version; } catch {}

// ── .env file loading ─────────────────────────────────────────────────────────
// Parses KEY=VALUE pairs from a .env string. Exported for unit-testing.
function parseDotEnv(content) {
  const vars = {};
  for (const raw of content.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    // Strip surrounding single or double quotes.
    if (val.length >= 2 &&
        ((val[0] === '"' && val[val.length - 1] === '"') ||
         (val[0] === "'" && val[val.length - 1] === "'")))
      val = val.slice(1, -1);
    vars[key] = val;
  }
  return vars;
}

// Load .env from cwd (and script dir if different) before reading any config.
// Variables already in the environment always take precedence (12-factor style).
;(function loadDotEnv() {
  const candidates = [path.join(process.cwd(), '.env')];
  if (path.resolve(__dirname) !== path.resolve(process.cwd()))
    candidates.push(path.join(__dirname, '.env'));
  for (const file of candidates) {
    let src;
    try { src = fs.readFileSync(file, 'utf8'); } catch { continue; }
    const vars = parseDotEnv(src);
    let loaded = 0;
    for (const [k, v] of Object.entries(vars)) {
      if (!(k in process.env)) { process.env[k] = v; loaded++; }
    }
    if (loaded > 0)
      console.log(`[claude-ollama-proxy] .env: loaded ${loaded} variable(s) from ${file}`);
    break; // stop after the first found file
  }
})();

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
// LOG_LEVEL controls verbosity. 'info' (default) logs one summary line per request.
// 'debug' additionally logs the full translated OpenAI-format body sent to Ollama and,
// for non-streaming requests, the raw Ollama response — invaluable for diagnosing why
// message conversion, system-prompt injection, or tool formatting produces unexpected results.
// Large base64 image payloads are automatically truncated so logs stay readable.
const LOG_LEVEL     = process.env.LOG_LEVEL        || 'info';
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
// Hard ceiling on output tokens per request. When set, any client-requested max_tokens above
// this value is silently clamped down to PROXY_HARD_MAX_TOKENS. Useful on shared deployments
// to prevent a single caller from monopolising the GPU with an enormous generation budget.
// The default PROXY_MAX_TOKENS is also capped so operators only need to set one value.
// Applies to POST /v1/messages, /v1/chat/completions, and /v1/completions.
const PROXY_HARD_MAX_TOKENS = (() => {
  if (!process.env.PROXY_HARD_MAX_TOKENS) return null;
  const n = Number(process.env.PROXY_HARD_MAX_TOKENS);
  if (!Number.isFinite(n) || n < 1) {
    console.warn('Warning: PROXY_HARD_MAX_TOKENS must be a positive integer, ignoring');
    return null;
  }
  return Math.floor(n);
})();
// Optional system prompt injected before every request's system field.
// Useful for enforcing consistent model behavior across all callers without modifying clients.
// When the client also supplies a system prompt, the proxy's prompt is prepended (separated by
// two newlines). For array-form system prompts the proxy text becomes the first content block.
const PROXY_SYSTEM_PROMPT = process.env.PROXY_SYSTEM_PROMPT || null;
// Optional hard body-size limit (bytes). Requests exceeding this via Content-Length are
// rejected with 413 before the body is read, protecting against runaway base64-image payloads.
// Default is no limit. Example: PROXY_MAX_BODY_SIZE=10485760 for 10 MB.
const PROXY_WARMUP       = process.env.PROXY_WARMUP === 'true';
// Optional limit on simultaneous in-flight Ollama inference requests.
// When reached, new requests get 503 overloaded_error + Retry-After: 1 instead of
// competing for GPU VRAM and causing OOM errors or severe latency spikes.
// Applies to POST /v1/messages, /v1/chat/completions, and /v1/completions.
// Embeddings and token-counting are not gated (they use separate lightweight endpoints).
const PROXY_MAX_CONCURRENCY = process.env.PROXY_MAX_CONCURRENCY
  ? Number(process.env.PROXY_MAX_CONCURRENCY) : null;
// Optional queue for requests that arrive when all concurrency slots are taken.
// PROXY_MAX_QUEUE_SIZE  — max number of requests that may wait in the queue; when full
//                         new arrivals still get 503 immediately. Default: no queuing (503 always).
// PROXY_MAX_QUEUE_TIMEOUT — ms a queued request waits before giving up with 503.
//                           Default: no timeout (waits indefinitely until a slot opens).
const PROXY_MAX_QUEUE_SIZE    = process.env.PROXY_MAX_QUEUE_SIZE
  ? Number(process.env.PROXY_MAX_QUEUE_SIZE) : null;
const PROXY_MAX_QUEUE_TIMEOUT = process.env.PROXY_MAX_QUEUE_TIMEOUT
  ? Number(process.env.PROXY_MAX_QUEUE_TIMEOUT) : null;
const PROXY_MAX_BODY_SIZE = (() => {
  if (!process.env.PROXY_MAX_BODY_SIZE) return null;
  const n = Number(process.env.PROXY_MAX_BODY_SIZE);
  if (!Number.isFinite(n) || n <= 0) {
    console.warn('Warning: PROXY_MAX_BODY_SIZE is not a valid positive number, ignoring');
    return null;
  }
  return n;
})();

// Optional JSON object of arbitrary Ollama model parameters to include in every request.
// Useful for deployment-level tuning (repeat_penalty, mirostat, num_gpu, tfs_z, etc.)
// without adding individual env vars for each knob. Per-request client params take
// precedence; dedicated env vars (OLLAMA_NUM_CTX, OLLAMA_KEEP_ALIVE) take highest precedence.
// Example: OLLAMA_OPTIONS='{"repeat_penalty":1.1,"mirostat":2,"num_gpu":33}'
function parseOllamaOptions(str) {
  if (!str) return {};
  let v;
  try { v = JSON.parse(str); } catch (e) {
    console.warn('Warning: OLLAMA_OPTIONS is not valid JSON, ignoring:', e.message);
    return {};
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) {
    console.warn('Warning: OLLAMA_OPTIONS must be a JSON object (not array/scalar), ignoring');
    return {};
  }
  return v;
}
const OLLAMA_OPTIONS = parseOllamaOptions(process.env.OLLAMA_OPTIONS);

// Returns a sanitized deep copy of obj with long base64 strings replaced by a short
// placeholder so debug log lines stay human-readable even when requests contain images.
// Matches: the `data` key for Anthropic base64 source blocks, and the `url` key when
// its value is a data-URL (i.e. starts with "data:") for OpenAI image_url blocks.
function sanitizeForLog(obj, maxChars = 200) {
  if (Array.isArray(obj)) return obj.map(x => sanitizeForLog(x, maxChars));
  if (obj && typeof obj === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === 'string' && v.length > maxChars &&
          (k === 'data' || (k === 'url' && v.startsWith('data:')))) {
        out[k] = `<base64 ${v.length} chars>`;
      } else {
        out[k] = sanitizeForLog(v, maxChars);
      }
    }
    return out;
  }
  return obj;
}

// Extracts a human-readable error string from an Ollama error response body.
// Ollama typically returns {"error":"..."} JSON; this avoids double-encoding that
// string as the `message` field and instead surfaces the inner error text directly.
// Falls back to the raw text if the body is not JSON or has no error/message key.
function parseOllamaError(text) {
  if (!text) return text;
  try {
    const obj = JSON.parse(text);
    if (obj && typeof obj.error === 'string') return obj.error;
    if (obj && typeof obj.message === 'string') return obj.message;
  } catch { /* not JSON — use raw text */ }
  return text;
}

// Logs obj as pretty JSON under a label when LOG_LEVEL=debug. No-op otherwise.
function debugLog(label, obj) {
  if (LOG_LEVEL !== 'debug') return;
  console.log(`[DEBUG] ${label}:\n${JSON.stringify(sanitizeForLog(obj), null, 2)}`);
}

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
  activeStreams:      0,
  activeLlmRequests: 0,  // in-flight inference requests (gated by PROXY_MAX_CONCURRENCY)
  queuedLlmRequests: 0,  // requests waiting for a concurrency slot
  errors:            0,
  modelsUsed:   {},   // 'model-name' → { requests, tokensIn, tokensOut }
};

// Queue of { onGranted } entries waiting for a concurrency slot.
const _concurrencyQueue = [];

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
    const rid = res.getHeader('request-id') ? ` id=${res.getHeader('request-id')}` : '';
    console.log(`${req.method} ${req.url} ${res.statusCode} ${ms}ms${toks}${rid}`);
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

// Resolves the effective max_tokens for a request:
//   1. Uses the client's value if provided and valid.
//   2. Falls back to PROXY_MAX_TOKENS when client omits it.
//   3. Clamps the result to PROXY_HARD_MAX_TOKENS (when set).
// Returns { value: number } on success, or { error: string } when the client
// supplied an invalid value (non-integer, ≤ 0) so the caller can return 400.
function resolveMaxTokens(clientValue) {
  if (clientValue !== undefined && clientValue !== null) {
    const n = Number(clientValue);
    if (!Number.isFinite(n) || n < 1 || Math.floor(n) !== n) {
      return { error: '`max_tokens` must be a positive integer' };
    }
    const effective = PROXY_HARD_MAX_TOKENS ? Math.min(n, PROXY_HARD_MAX_TOKENS) : n;
    return { value: effective };
  }
  const fallback = PROXY_HARD_MAX_TOKENS
    ? Math.min(PROXY_MAX_TOKENS, PROXY_HARD_MAX_TOKENS)
    : PROXY_MAX_TOKENS;
  return { value: fallback };
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
      // OpenAI tool messages only support string content, so images are collected separately
      // and appended as a follow-up user message so vision-capable models can still see them.
      const pendingImages = [];
      for (const tr of toolResults) {
        if (!tr.tool_use_id) continue;
        const rawContent = Array.isArray(tr.content)
          ? tr.content.map(c => {
              if (c.type === 'document') return documentBlockToText(c) || '';
              if (c.type === 'image') return '';
              return c.text || '';
            }).join('')
          : (tr.content || '');
        const content = tr.is_error ? `[ERROR] ${rawContent}` : rawContent;
        result.push({ role: 'tool', tool_call_id: tr.tool_use_id, content });
        if (Array.isArray(tr.content)) {
          for (const c of tr.content) {
            if (c.type === 'image') {
              const img = imageBlockToOpenAI(c);
              if (img) pendingImages.push(img);
            }
          }
        }
      }
      // Merge any trailing text parts and tool-result images into one user message.
      const followUpParts = [
        ...(textParts.length > 0 ? [{ type: 'text', text: textParts.map(b => b.text).join('') }] : []),
        ...pendingImages,
      ];
      if (followUpParts.length > 0) {
        result.push({
          role: 'user',
          content: followUpParts.length === 1 && followUpParts[0].type === 'text'
            ? followUpParts[0].text
            : followUpParts,
        });
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
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
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

// Transfers a concurrency slot to the first queued waiter, or decrements the in-flight
// counter if the queue is empty. Called by trackActiveLlmRequest on request completion.
function releaseLlmSlot() {
  if (_concurrencyQueue.length > 0) {
    // Hand the slot directly to the next waiter — activeLlmRequests stays the same.
    const { onGranted } = _concurrencyQueue.shift();
    onGranted();
  } else {
    _metrics.activeLlmRequests--;
  }
}

// Async slot acquisition with optional queuing when at max concurrency.
// Returns true when the caller has acquired a slot (activeLlmRequests already incremented).
// Returns false and writes a 503 / queued-too-long response when no slot can be obtained.
// When PROXY_MAX_CONCURRENCY is unset, always grants immediately and tracks the counter
// (fixes a prior bug where activeLlmRequests went negative when no limit was configured).
async function acquireLlmSlot(req, res) {
  if (!PROXY_MAX_CONCURRENCY) {
    _metrics.activeLlmRequests++;
    return true;
  }
  if (_metrics.activeLlmRequests < PROXY_MAX_CONCURRENCY) {
    _metrics.activeLlmRequests++;
    return true;
  }
  // At max concurrency — try to queue if PROXY_MAX_QUEUE_SIZE is set.
  if (!PROXY_MAX_QUEUE_SIZE || _concurrencyQueue.length >= PROXY_MAX_QUEUE_SIZE) {
    const queueInfo = PROXY_MAX_QUEUE_SIZE
      ? `, queue full (${PROXY_MAX_QUEUE_SIZE})`
      : '';
    res.setHeader('retry-after', '1');
    if (!res.headersSent) res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      type: 'error',
      error: {
        type: 'overloaded_error',
        message: `Proxy at max concurrency (${PROXY_MAX_CONCURRENCY} in-flight${queueInfo}). Retry after 1s.`,
      },
    }));
    return false;
  }

  // Queue the request and suspend until a slot is handed to us.
  _metrics.queuedLlmRequests++;
  return new Promise(resolve => {
    let done = false;

    // Called by releaseLlmSlot when our turn arrives. The slot is ours;
    // activeLlmRequests was NOT decremented (transferred, not freed+re-acquired).
    const onGranted = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      req.socket.off('close', onDisconnect);
      _metrics.queuedLlmRequests--;
      resolve(true);
    };

    // Called on timeout or client disconnect while waiting in the queue.
    const onAbort = (reason) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      req.socket.off('close', onDisconnect);
      const idx = _concurrencyQueue.findIndex(e => e.onGranted === onGranted);
      if (idx >= 0) _concurrencyQueue.splice(idx, 1);
      _metrics.queuedLlmRequests--;
      if (!res.headersSent) {
        res.setHeader('retry-after', '1');
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ type: 'error', error: { type: 'overloaded_error', message: reason } }));
      }
      resolve(false);
    };

    const timer = PROXY_MAX_QUEUE_TIMEOUT
      ? setTimeout(() => onAbort(`Request queued for ${PROXY_MAX_QUEUE_TIMEOUT}ms with no slot available. Retry after 1s.`), PROXY_MAX_QUEUE_TIMEOUT)
      : null;
    const onDisconnect = () => onAbort('Client disconnected while waiting in concurrency queue.');

    req.socket.once('close', onDisconnect);
    _concurrencyQueue.push({ onGranted });
  });
}

// Returns true and records the in-flight request if below PROXY_MAX_CONCURRENCY;
// writes a 503 overloaded_error and returns false when the limit is reached.
// Node.js is single-threaded, so the check + increment before the first await is race-free.
function checkConcurrency(res) {
  if (!PROXY_MAX_CONCURRENCY) return true;
  if (_metrics.activeLlmRequests >= PROXY_MAX_CONCURRENCY) {
    res.setHeader('retry-after', '1');
    if (!res.headersSent) res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      type: 'error',
      error: {
        type: 'overloaded_error',
        message: `Proxy at max concurrency (${PROXY_MAX_CONCURRENCY} in-flight requests). Retry after 1s.`,
      },
    }));
    return false;
  }
  _metrics.activeLlmRequests++;
  return true;
}

// Schedules exactly one slot release when the response ends (normal finish or dropped connection).
// Calls releaseLlmSlot which either hands the slot to the next queue waiter or decrements the counter.
function trackActiveLlmRequest(res) {
  let done = false;
  const dec = () => { if (!done) { done = true; releaseLlmSlot(); } };
  res.once('finish', dec);
  res.once('close', dec);
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

  const maxTokensResult = resolveMaxTokens(anthropicReq.max_tokens);
  if (maxTokensResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: maxTokensResult.error } }));
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
    max_tokens: maxTokensResult.value,
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
  // OLLAMA_OPTIONS: fill in Ollama-specific params not already set by the request.
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS)) {
    if (!(k in openaiReq)) openaiReq[k] = v;
  }
  // Dedicated env vars take highest precedence (unconditional overwrite).
  if (OLLAMA_NUM_CTX)    openaiReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) openaiReq.keep_alive = OLLAMA_KEEP_ALIVE;

  debugLog(`→ Ollama [${ollamaBase}]`, openaiReq);

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
    const errText = await ollamaRes.text();
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_error', message: parseOllamaError(errText) } }));
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
    debugLog('← Ollama response', data);
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
  let textBlockIdx  = -1;      // anthropic index of the currently open text block
  const toolBlocks  = {};      // openai tool index → { anthropicIndex, id, name, args }
  let inputTokens   = 0;
  let outputTokens  = 0;
  let stopReason    = null;    // set on finish_reason; message_delta deferred until after loop

  // State machine for routing <think>…</think> content to Anthropic thinking blocks.
  // Supports interleaved thinking: multiple alternating think/text blocks in one response
  // (Qwen3-thinking, DeepSeek-R1, and models using anthropic-beta: interleaved-thinking-*).
  let thinkState   = 'initial'; // 'initial' | 'thinking' | 'text'
  let thinkBuf     = '';        // chars pending routing (straddle tag boundaries)
  let thinkCount   = 0;         // thinking blocks fully closed so far
  let thinkIdx     = 0;         // anthropic index of the currently open thinking block
  let nextBlockIdx = 0;         // monotonic block-index counter shared by think/text/tool blocks

  function routeThinkChunk(flush) {
    for (;;) {
      if (!thinkBuf.length) break;

      if (thinkState === 'initial') {
        const tag = '<think>';
        if (thinkBuf.startsWith(tag)) {
          thinkIdx = nextBlockIdx++;
          sendSSE(res, 'content_block_start', {
            type: 'content_block_start', index: thinkIdx,
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
            type: 'content_block_delta', index: thinkIdx,
            delta: { type: 'thinking_delta', thinking: thinkBuf.slice(0, ei) }
          });
          sendSSE(res, 'content_block_delta', {
            type: 'content_block_delta', index: thinkIdx,
            delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
          });
          sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkIdx });
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
            type: 'content_block_delta', index: thinkIdx,
            delta: { type: 'thinking_delta', thinking: thinkBuf.slice(0, safe) }
          });
          thinkBuf = thinkBuf.slice(safe);
        }
        break;
      }

      // thinkState === 'text'
      // Check for a new <think> tag — supports interleaved thinking/text blocks
      // (anthropic-beta: interleaved-thinking-*, Qwen3-thinking, DeepSeek-R1).
      const tag = '<think>';
      const ti = thinkBuf.indexOf(tag);
      if (ti === 0) {
        // Buffer starts with <think>: close open text block and switch back to 'initial'.
        if (textBlockOpen) {
          sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textBlockIdx });
          textBlockOpen = false;
        }
        thinkState = 'initial';
        continue;
      }
      if (ti > 0) {
        // Text before <think>: emit it, close text block, then let 'initial' open the think block.
        if (!textBlockOpen) {
          textBlockIdx = nextBlockIdx++;
          sendSSE(res, 'content_block_start', {
            type: 'content_block_start', index: textBlockIdx,
            content_block: { type: 'text', text: '' }
          });
          sendSSE(res, 'ping', { type: 'ping' });
          textBlockOpen = true;
        }
        sendSSE(res, 'content_block_delta', {
          type: 'content_block_delta', index: textBlockIdx,
          delta: { type: 'text_delta', text: thinkBuf.slice(0, ti) }
        });
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textBlockIdx });
        textBlockOpen = false;
        thinkBuf = thinkBuf.slice(ti);
        thinkState = 'initial';
        continue;
      }
      // No '<think>' in buffer. Hold back a possible partial tag at the end (unless flushing).
      if (!flush) {
        const lt = thinkBuf.lastIndexOf('<');
        if (lt >= 0 && tag.startsWith(thinkBuf.slice(lt))) {
          const safe = thinkBuf.slice(0, lt);
          if (safe) {
            if (!textBlockOpen) {
              textBlockIdx = nextBlockIdx++;
              sendSSE(res, 'content_block_start', {
                type: 'content_block_start', index: textBlockIdx,
                content_block: { type: 'text', text: '' }
              });
              sendSSE(res, 'ping', { type: 'ping' });
              textBlockOpen = true;
            }
            sendSSE(res, 'content_block_delta', {
              type: 'content_block_delta', index: textBlockIdx,
              delta: { type: 'text_delta', text: safe }
            });
          }
          thinkBuf = thinkBuf.slice(lt);
          break;
        }
      }
      // No think tags — emit everything as a text delta.
      if (!textBlockOpen) {
        textBlockIdx = nextBlockIdx++;
        sendSSE(res, 'content_block_start', {
          type: 'content_block_start', index: textBlockIdx,
          content_block: { type: 'text', text: '' }
        });
        sendSSE(res, 'ping', { type: 'ping' });
        textBlockOpen = true;
      }
      sendSSE(res, 'content_block_delta', {
        type: 'content_block_delta', index: textBlockIdx,
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
              const ai = nextBlockIdx + oi;
              // Bump nextBlockIdx past this tool block so any subsequent text/thinking
              // block gets a higher index and cannot clash with tool block indices.
              nextBlockIdx = Math.max(nextBlockIdx, ai + 1);
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
              type: 'content_block_delta', index: thinkIdx,
              delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
            });
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkIdx });
            thinkCount++;
          }
          if (textBlockOpen) {
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textBlockIdx });
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
          type: 'content_block_delta', index: thinkIdx,
          delta: { type: 'signature_delta', signature: 'ollama-proxy-extracted' }
        });
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: thinkIdx });
        thinkCount++;
      }
      if (textBlockOpen) {
        sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: textBlockIdx });
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

  const ollamaModels = (data.models || []).map(m => ({
    id: m.name,
    object: 'model',
    created: m.modified_at ? Math.floor(new Date(m.modified_at).getTime() / 1000) : 0,
    owned_by: 'ollama',
    // Expose Ollama metadata already present in /api/tags at no extra cost.
    // Clients (Continue, Open WebUI, Cursor) can use parameter_size and
    // quantization_level to display helpful model info without extra requests.
    ...(m.details ? { details: m.details } : {}),
    ...(m.size != null ? { size: m.size } : {}),
  }));

  // When MODEL_MAP is configured, also expose the Claude alias names so model-picker
  // clients (Cursor, Continue, OpenWebUI) can discover and select them. Aliases that
  // clash with an actual Ollama model ID are skipped to avoid duplicates.
  const ollamaIds = new Set(ollamaModels.map(m => m.id));
  const createdByTarget = {};
  for (const m of data.models || []) {
    createdByTarget[m.name] = m.modified_at ? Math.floor(new Date(m.modified_at).getTime() / 1000) : 0;
  }
  const byTargetDetails = {};
  for (const m of data.models || []) {
    byTargetDetails[m.name] = { details: m.details, size: m.size };
  }
  const aliasModels = Object.entries(MODEL_MAP)
    .filter(([alias]) => !ollamaIds.has(alias))
    .map(([alias, target]) => {
      const meta = byTargetDetails[target] || {};
      return {
        id: alias,
        object: 'model',
        created: createdByTarget[target] || 0,
        owned_by: 'ollama',
        ...(meta.details ? { details: meta.details } : {}),
        ...(meta.size != null ? { size: meta.size } : {}),
      };
    });

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ object: 'list', data: [...ollamaModels, ...aliasModels] }));
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

  const allModels = data.models || [];

  // Try exact match against real Ollama model names first.
  let model = allModels.find(m => m.name === modelId);

  // If not found, resolve via MODEL_MAP aliases — same logic as handleModels exposes them
  // in GET /v1/models. This ensures GET /v1/models/:id is consistent with GET /v1/models.
  if (!model) {
    let targetName = MODEL_MAP[modelId];
    if (!targetName) {
      for (const [key, target] of Object.entries(MODEL_MAP)) {
        if (modelId.startsWith(key)) { targetName = target; break; }
      }
    }
    if (targetName) model = allModels.find(m => m.name === targetName);
  }

  if (!model) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Model '${modelId}' not found in Ollama` } }));
    return;
  }

  // Best-effort: enrich with /api/show details (context length, template, default system).
  // If the show call fails for any reason the basic response is still returned.
  let showData = null;
  try {
    const showRes = await fetch(`${ollamaBase}/api/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: model.name }),
      signal: AbortSignal.timeout(5000),
    });
    if (showRes.ok) showData = await showRes.json();
  } catch { /* ignore — show is best-effort */ }

  const resp = {
    id: modelId,   // always return the requested ID (alias or real name)
    object: 'model',
    created: model.modified_at ? Math.floor(new Date(model.modified_at).getTime() / 1000) : 0,
    owned_by: 'ollama',
    ...(model.details ? { details: model.details } : {}),
    ...(model.size != null ? { size: model.size } : {}),
  };

  if (showData) {
    // model_info keys follow GGUF naming conventions; fall back gracefully if absent.
    const ctxLen = showData.model_info?.['llm.context_length'] ?? null;
    if (ctxLen != null) resp.context_length = ctxLen;
    if (showData.system)   resp.system   = showData.system;
    if (showData.template) resp.template  = showData.template;
  }

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(resp));
}

// DELETE /v1/models/:id — deletes a model from Ollama via DELETE /api/delete.
// Auth-gated like other model management endpoints. Returns {deleted:true,id} on
// success, 404 if the model isn't in Ollama, 502 if Ollama is unreachable.
async function handleDeleteModel(req, res, modelId) {
  const ollamaBase = getOllamaHost();
  let ollamaRes;
  try {
    ollamaRes = await fetch(`${ollamaBase}/api/delete`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId }),
      signal: AbortSignal.timeout(10_000),
    });
  } catch (e) {
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: { type: 'ollama_unreachable', message: e.message + (isConnRefused ? ' — is Ollama running?' : '') }
    }));
    return;
  }

  if (ollamaRes.status === 200) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ deleted: true, id: modelId }));
  } else if (ollamaRes.status === 404) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Model '${modelId}' not found in Ollama` } }));
  } else {
    const errText = await ollamaRes.text().catch(() => '');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: { type: 'ollama_error', message: parseOllamaError(errText) || `Ollama returned HTTP ${ollamaRes.status}` }
    }));
  }
}

// POST /v1/models/pull — pulls a model from the Ollama registry through the proxy.
// Auth-gated like other model management endpoints. Supports non-streaming
// ({pulled:true,id,object:'model'} on success) and streaming (Ollama's NDJSON
// progress piped as SSE so operators can watch download progress in real time).
// Client-abort propagation cancels in-flight pulls, freeing bandwidth immediately.
async function handlePullModel(req, res) {
  const ollamaBase = getOllamaHost();

  const body = await readBody(req);
  let pullReq;
  try { pullReq = JSON.parse(body); }
  catch {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (!pullReq.model || typeof pullReq.model !== 'string') {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`model` is required and must be a string' } }));
    return;
  }

  const streaming = pullReq.stream === true;

  const ac = new AbortController();
  const onClientClose = () => { if (!res.writableEnded) ac.abort(); };
  req.socket.once('close', onClientClose);

  let ollamaRes;
  try {
    ollamaRes = await fetch(`${ollamaBase}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: pullReq.model, stream: streaming }),
      signal: ac.signal,
    });
  } catch (e) {
    req.socket.off('close', onClientClose);
    if (e.name === 'AbortError') { if (!res.writableEnded) res.end(); return; }
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: { type: 'ollama_unreachable', message: e.message + (isConnRefused ? ' — is Ollama running? Try: ollama serve' : '') }
    }));
    return;
  }

  if (!ollamaRes.ok) {
    req.socket.off('close', onClientClose);
    const errText = await ollamaRes.text().catch(() => '');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_error', message: parseOllamaError(errText) || `Ollama returned HTTP ${ollamaRes.status}` } }));
    return;
  }

  if (!streaming) {
    let data;
    try { data = await ollamaRes.json(); }
    catch {
      req.socket.off('close', onClientClose);
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'ollama_error', message: 'Failed to parse Ollama pull response' } }));
      return;
    }
    req.socket.off('close', onClientClose);
    if (data.error) {
      res.writeHead(422, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'ollama_error', message: data.error } }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ id: pullReq.model, pulled: true, object: 'model', status: data.status || 'success' }));
    return;
  }

  // Streaming: pipe Ollama's NDJSON progress as SSE so callers can watch download progress.
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let success = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let obj;
        try { obj = JSON.parse(line); } catch { continue; }
        if (obj.status === 'success' || obj.status === 'already exists') success = true;
        res.write(`data: ${JSON.stringify(obj)}\n\n`);
      }
    }
    if (!res.writableEnded) {
      res.write(`data: ${JSON.stringify({ id: pullReq.model, pulled: success, done: true })}\n\n`);
    }
  } catch (e) {
    if (e.name !== 'AbortError' && !res.writableEnded) {
      res.write(`data: ${JSON.stringify({ error: { type: 'stream_error', message: e.message } })}\n\n`);
    }
  } finally {
    req.socket.off('close', onClientClose);
  }

  res.end();
}

// ── Anthropic Messages Batch API ─────────────────────────────────────────────
// In-memory store. Each batch: { id, status, created_at, expires_at, ended_at,
//   cancel_initiated_at, requests, results (Map custom_id→result), cancelRequested }
// status: 'in_progress' | 'canceling' | 'ended'
const _batches = new Map();

function newBatchId() {
  return 'msgbatch_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function batchRequestCounts(batch) {
  let processing = 0, succeeded = 0, errored = 0, canceled = 0, expired = 0;
  for (const req of batch.requests) {
    const r = batch.results.get(req.custom_id);
    if (!r)                    { processing++; continue; }
    if (r.type === 'succeeded') succeeded++;
    else if (r.type === 'errored')  errored++;
    else if (r.type === 'canceled') canceled++;
    else if (r.type === 'expired')  expired++;
  }
  return { processing, succeeded, errored, canceled, expired };
}

function batchToResponse(batch, baseUrl) {
  const counts = batchRequestCounts(batch);
  return {
    id:                   batch.id,
    type:                 'message_batch',
    processing_status:    batch.status,
    request_counts:       counts,
    ended_at:             batch.ended_at             || null,
    created_at:           batch.created_at,
    expires_at:           batch.expires_at,
    cancel_initiated_at:  batch.cancel_initiated_at  || null,
    results_url: batch.status === 'ended'
      ? `${baseUrl}/v1/messages/batches/${batch.id}/results`
      : null,
  };
}

// Process a single batch request item — reuses the same conversion logic as
// handleMessages but operates synchronously against Ollama (non-streaming).
// Returns { type: 'succeeded', message } or { type: 'errored', error }.
async function processBatchRequest(anthropicReq, ollamaBase) {
  const effectiveModel = resolveModel(anthropicReq.model);
  const maxTokensResult = resolveMaxTokens(anthropicReq.max_tokens);
  if (maxTokensResult.error)
    return { type: 'errored', error: { type: 'invalid_request_error', message: maxTokensResult.error } };

  const openaiReq = {
    model:      effectiveModel,
    messages:   toOpenAIMessages(anthropicReq.messages, injectSystemPrompt(anthropicReq.system)),
    stream:     false,
    max_tokens: maxTokensResult.value,
  };
  const tools = toOpenAITools(anthropicReq.tools);
  if (tools) openaiReq.tools = tools;
  const toolChoice = toOpenAIToolChoice(anthropicReq.tool_choice);
  if (toolChoice !== undefined) openaiReq.tool_choice = toolChoice;
  if (anthropicReq.temperature          !== undefined) openaiReq.temperature          = anthropicReq.temperature;
  if (anthropicReq.top_p                !== undefined) openaiReq.top_p                = anthropicReq.top_p;
  if (anthropicReq.top_k                !== undefined) openaiReq.top_k                = anthropicReq.top_k;
  if (anthropicReq.seed                 !== undefined) openaiReq.seed                 = anthropicReq.seed;
  if (anthropicReq.stop_sequences?.length)             openaiReq.stop                 = anthropicReq.stop_sequences;
  if (anthropicReq.disable_parallel_tool_use === true) openaiReq.parallel_tool_calls  = false;
  if (anthropicReq.thinking?.type === 'enabled')       openaiReq.think                = true;
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS))
    if (!(k in openaiReq)) openaiReq[k] = v;
  if (OLLAMA_NUM_CTX)    openaiReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) openaiReq.keep_alive = OLLAMA_KEEP_ALIVE;

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/v1/chat/completions`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(openaiReq),
    });
  } catch (e) {
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    return { type: 'errored', error: {
      type: 'ollama_unreachable',
      message: e.message + (isConnRefused ? ' — is Ollama running?' : ''),
    }};
  }

  if (!ollamaRes.ok) {
    const errText = await ollamaRes.text().catch(() => '');
    return { type: 'errored', error: { type: 'ollama_error', message: parseOllamaError(errText) } };
  }

  let data;
  try { data = await ollamaRes.json(); }
  catch { return { type: 'errored', error: { type: 'ollama_error', message: 'Failed to parse Ollama response' } }; }

  const choice = data.choices?.[0];
  if (!choice)
    return { type: 'errored', error: { type: 'ollama_error', message: 'Empty choices in Ollama response' } };

  const msg     = choice.message;
  const content = [];
  if (msg.content) {
    const thinkParts = extractThinkingParts(msg.content);
    if (thinkParts) {
      for (const p of thinkParts) {
        content.push(p.type === 'thinking'
          ? { type: 'thinking', thinking: p.thinking, signature: 'ollama-proxy-extracted' }
          : { type: 'text', text: p.text });
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

  const promptTok     = data.usage?.prompt_tokens     || 0;
  const completionTok = data.usage?.completion_tokens || 0;
  recordTokens(promptTok, completionTok, effectiveModel);

  return {
    type: 'succeeded',
    message: {
      id:            newMsgId(),
      type:          'message',
      role:          'assistant',
      content,
      model:         effectiveModel,
      stop_reason:   choice.finish_reason === 'tool_calls' ? 'tool_use'
                   : choice.finish_reason === 'length'     ? 'max_tokens'
                   : 'end_turn',
      stop_sequence: null,
      usage: {
        input_tokens:               promptTok,
        output_tokens:              completionTok,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens:     0,
      },
    },
  };
}

// Drive a batch to completion in the background (serially to avoid GPU OOM).
// Runs as a fire-and-forget task started with setImmediate from handleCreateBatch.
async function processBatch(batch) {
  const ollamaBase = getOllamaHost();
  for (const item of batch.requests) {
    if (batch.cancelRequested) {
      batch.results.set(item.custom_id, { type: 'canceled' });
      continue;
    }
    const result = await processBatchRequest(item.params, ollamaBase).catch(e => ({
      type: 'errored',
      error: { type: 'internal_error', message: e.message },
    }));
    batch.results.set(item.custom_id, result);
  }
  batch.status   = 'ended';
  batch.ended_at = new Date().toISOString();
}

function getBatchBaseUrl(req) {
  const proto = req.socket?.encrypted ? 'https' : 'http';
  const host  = req.headers.host || `localhost:${PORT}`;
  return `${proto}://${host}`;
}

async function handleCreateBatch(req, res) {
  const body = await readBody(req);
  let batchReq;
  try { batchReq = JSON.parse(body); }
  catch {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (!Array.isArray(batchReq.requests) || batchReq.requests.length === 0) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`requests` must be a non-empty array' } }));
    return;
  }

  for (const r of batchReq.requests) {
    if (!r.custom_id || typeof r.custom_id !== 'string') {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Each request must have a string `custom_id`' } }));
      return;
    }
    if (!r.params || !Array.isArray(r.params.messages)) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}' must have a params.messages array` } }));
      return;
    }
  }

  const now   = new Date();
  const batch = {
    id:                  newBatchId(),
    status:              'in_progress',
    created_at:          now.toISOString(),
    expires_at:          new Date(now.getTime() + 24 * 60 * 60 * 1000).toISOString(),
    ended_at:            null,
    cancel_initiated_at: null,
    requests:            batchReq.requests,
    results:             new Map(),
    cancelRequested:     false,
  };

  _batches.set(batch.id, batch);
  setImmediate(() => processBatch(batch));

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

async function handleListBatches(req, res) {
  const baseUrl = getBatchBaseUrl(req);
  const all     = [..._batches.values()].reverse();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    data:     all.map(b => batchToResponse(b, baseUrl)),
    has_more: false,
    first_id: all[0]?.id                  || null,
    last_id:  all[all.length - 1]?.id     || null,
  }));
}

async function handleGetBatch(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Batch '${batchId}' not found` } }));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

async function handleGetBatchResults(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Batch '${batchId}' not found` } }));
    return;
  }
  if (batch.status !== 'ended') {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Batch '${batchId}' has not ended yet (status: ${batch.status})` } }));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/x-jsonl' });
  for (const item of batch.requests) {
    const result = batch.results.get(item.custom_id)
      || { type: 'errored', error: { type: 'internal_error', message: 'No result recorded' } };
    res.write(JSON.stringify({ custom_id: item.custom_id, result }) + '\n');
  }
  res.end();
}

async function handleCancelBatch(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Batch '${batchId}' not found` } }));
    return;
  }
  if (batch.status === 'ended') {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Batch '${batchId}' has already ended` } }));
    return;
  }
  batch.cancelRequested    = true;
  batch.cancel_initiated_at = new Date().toISOString();
  if (batch.status !== 'ended') batch.status = 'canceling';
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

async function handleEmbeddings(req, res) {
  const ollamaBase = getOllamaHost();
  const ac = new AbortController();

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
  let embedReq;
  try { embedReq = JSON.parse(body); }
  catch {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (embedReq.input == null) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`input` is required' } }));
    return;
  }

  const effectiveModel = resolveModel(embedReq.model);

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...OLLAMA_OPTIONS, model: effectiveModel, input: embedReq.input }),
      signal: ac.signal,
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
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_unreachable', message: e.message + (isConnRefused ? ' — is Ollama running? Try: ollama serve' : '') } }));
    return;
  }

  req.socket.off('close', onClientClose);
  clearTO();

  if (!ollamaRes.ok) {
    const errText = await ollamaRes.text();
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_error', message: parseOllamaError(errText) } }));
    return;
  }

  let data;
  try { data = await ollamaRes.json(); }
  catch {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'ollama_error', message: 'Failed to parse Ollama embeddings response' } }));
    return;
  }

  const embeddings = data.embeddings || [];
  const promptTokens = data.prompt_eval_count || 0;

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    object: 'list',
    data: embeddings.map((emb, i) => ({ object: 'embedding', embedding: emb, index: i })),
    model: effectiveModel,
    usage: { prompt_tokens: promptTokens, total_tokens: promptTokens },
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
    version: PROXY_VERSION,
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
    active_llm_requests: _metrics.activeLlmRequests,
    queued_llm_requests: _metrics.queuedLlmRequests,
    concurrency_limit:   PROXY_MAX_CONCURRENCY,
    queue_limit:         PROXY_MAX_QUEUE_SIZE,
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

  out.push('# HELP proxy_active_llm_requests Current number of in-flight LLM inference requests');
  out.push('# TYPE proxy_active_llm_requests gauge');
  out.push(`proxy_active_llm_requests ${_metrics.activeLlmRequests}`);
  out.push('');

  out.push('# HELP proxy_queued_llm_requests Current number of LLM requests waiting in the concurrency queue');
  out.push('# TYPE proxy_queued_llm_requests gauge');
  out.push(`proxy_queued_llm_requests ${_metrics.queuedLlmRequests}`);
  out.push('');

  if (PROXY_MAX_CONCURRENCY) {
    out.push('# HELP proxy_concurrency_limit Configured max concurrent in-flight LLM requests');
    out.push('# TYPE proxy_concurrency_limit gauge');
    out.push(`proxy_concurrency_limit ${PROXY_MAX_CONCURRENCY}`);
    out.push('');
  }

  if (PROXY_MAX_QUEUE_SIZE) {
    out.push('# HELP proxy_queue_limit Configured max number of requests that may wait in the concurrency queue');
    out.push('# TYPE proxy_queue_limit gauge');
    out.push(`proxy_queue_limit ${PROXY_MAX_QUEUE_SIZE}`);
    out.push('');
  }

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

// GET / — live HTML dashboard showing proxy health, metrics, and model usage.
// Self-contained: zero external deps, polls /health + /metrics every 5 s.
// Static config (model, port, hosts, etc.) is embedded server-side for instant display.
function handleDashboard(req, res) {
  const cfg = JSON.stringify({
    model:              MODEL,
    version:            PROXY_VERSION,
    port:               Number(PORT),
    hosts:              OLLAMA_HOSTS,
    auth:               !!PROXY_API_KEY,
    tls:                !!TLS_CERT,
    logFormat:          LOG_FORMAT,
    logLevel:           LOG_LEVEL,
    maxTokens:          PROXY_MAX_TOKENS,
    hardMaxTokens:      PROXY_HARD_MAX_TOKENS,
    numCtx:             OLLAMA_NUM_CTX,
    keepAlive:          OLLAMA_KEEP_ALIVE,
    rateLimitRpm:       RATE_LIMIT_RPM,
    rateLimitPerIpRpm:  RATE_LIMIT_PER_IP_RPM,
    warmup:             PROXY_WARMUP,
    timeout:            PROXY_TIMEOUT,
    maxBodySize:        PROXY_MAX_BODY_SIZE,
    systemPrompt:       PROXY_SYSTEM_PROMPT ? PROXY_SYSTEM_PROMPT.slice(0, 120) + (PROXY_SYSTEM_PROMPT.length > 120 ? '…' : '') : null,
    ollamaOptions:      Object.keys(OLLAMA_OPTIONS).length > 0 ? JSON.stringify(OLLAMA_OPTIONS) : null,
    maxConcurrency:     PROXY_MAX_CONCURRENCY,
    maxQueueSize:       PROXY_MAX_QUEUE_SIZE,
    queueTimeoutMs:     PROXY_MAX_QUEUE_TIMEOUT,
  });

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude-Ollama Proxy</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px 24px;min-height:100vh}
h1{color:#e6edf3;font-size:1.4rem;font-weight:600;padding-bottom:12px;border-bottom:1px solid #21262d;display:flex;align-items:center;gap:8px}
h1 span.dot{width:10px;height:10px;border-radius:50%;background:#3fb950;display:inline-block;flex-shrink:0}
h1 span.dot.err{background:#f85149}
.ts{color:#8b949e;font-size:11px;margin:8px 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px 16px}
.card h2{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:#8b949e;margin-bottom:10px}
.row{display:flex;justify-content:space-between;align-items:baseline;padding:3px 0;font-size:13px;border-bottom:1px solid #0d1117}
.row:last-child{border-bottom:none}
.lbl{color:#8b949e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:56%;font-size:12px}
.val{font-family:"SF Mono",Consolas,monospace;color:#58a6ff;text-align:right;font-size:12px}
.ok{color:#3fb950}.err{color:#f85149}.warn{color:#d29922}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.badge.ok{background:#0d3d1a;color:#3fb950}
.badge.err{background:#4d1b1b;color:#f85149}
.badge.warn{background:#3d2c00;color:#d29922}
.sep{margin-top:8px;padding-top:8px;border-top:1px dashed #21262d}
a{color:#58a6ff;text-decoration:none;font-size:12px}a:hover{text-decoration:underline}
</style>
</head>
<body>
<h1><span class="dot" id="dot"></span>Claude-Ollama Proxy</h1>
<div class="ts" id="ts">Loading…</div>
<div class="grid" id="grid"></div>
<script>
const C=${cfg};
function fmt(n){return n==null?'—':typeof n==='number'?n.toLocaleString():n}
function ms(n){return n==null?'—':n.toLocaleString()+'&thinsp;ms'}
function row(l,v,cls){return'<div class="row"><span class="lbl">'+l+'</span><span class="val '+(cls||'")>'+v+'</span></div>'}
function badge(ok,okT,errT){return'<span class="badge '+(ok?'ok':'err')+'">'+(ok?okT:errT)+'</span>'}
async function refresh(){
  const [h,m]=await Promise.all([
    fetch('/health').then(r=>r.json()).catch(()=>null),
    fetch('/metrics').then(r=>r.json()).catch(()=>null)
  ]);
  const ollamaOk=h&&(h.status==='ok'||h.ollama==='reachable');
  document.getElementById('dot').className='dot'+(ollamaOk?'':' err');
  document.getElementById('ts').textContent='Last updated: '+new Date().toLocaleTimeString()+' — refreshes every 5 s';
  let g='';

  // ── Status card ──────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Status</h2>';
  g+=row('Proxy',badge(true,'Running',''));
  g+=row('Ollama',badge(ollamaOk,'Reachable','Unreachable'));
  if(h&&h.hosts&&h.hosts.length>1){
    h.hosts.forEach(hh=>{
      const ok2=hh.status==='ok';
      g+=row(hh.url.replace(/^https?:\/\//,''),badge(ok2,'OK',hh.error||'Err'));
    });
  } else if(h&&h.ollamaError){
    g+=row('Error','<span class="err">'+h.ollamaError+'</span>');
  }
  g+=row('Active streams','<span class="'+(m&&m.active_streams>0?'ok':'val')+'">'+(m?m.active_streams:0)+'</span>');
  if(C.maxConcurrency){const cur=m?m.active_llm_requests:0;const cls=cur>=C.maxConcurrency?'warn':cur>0?'ok':'val';g+=row('In-flight requests','<span class="'+cls+'">'+cur+'/'+C.maxConcurrency+'</span>');}
  if(C.maxQueueSize){const q=m?m.queued_llm_requests:0;const cls=q>0?'warn':'val';g+=row('Queued requests','<span class="'+cls+'">'+q+'/'+C.maxQueueSize+'</span>');}
  g+=row('Uptime',m?fmt(m.uptime_seconds)+' s':'—');
  g+='</div>';

  // ── Config card ──────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Config</h2>';
  g+=row('Model',C.model);
  g+=row('Version',C.version);
  g+=row('Port',C.port);
  if(C.hosts.length===1)g+=row('Ollama host',C.hosts[0].replace(/^https?:\/\//,''));
  else g+=row('Ollama hosts',C.hosts.length+' (round-robin)');
  g+=row('Auth',badge(C.auth,'Enabled','Open — no key'));
  g+=row('TLS',badge(C.tls,'HTTPS','HTTP'));
  g+=row('Default max_tokens',fmt(C.maxTokens));
  if(C.hardMaxTokens)g+=row('Hard max_tokens cap',fmt(C.hardMaxTokens));
  g+=row('Context (num_ctx)',C.numCtx?fmt(C.numCtx):'model default');
  if(C.keepAlive)g+=row('Keep-alive',C.keepAlive);
  if(C.timeout)g+=row('Timeout',fmt(C.timeout)+' ms');
  if(C.rateLimitRpm)g+=row('Rate limit (global)',fmt(C.rateLimitRpm)+' req/min');
  if(C.rateLimitPerIpRpm)g+=row('Rate limit (per-IP)',fmt(C.rateLimitPerIpRpm)+' req/min');
  if(C.maxConcurrency)g+=row('Max concurrency',fmt(C.maxConcurrency)+' req');
  if(C.maxQueueSize)g+=row('Queue depth',fmt(C.maxQueueSize)+' req'+(C.queueTimeoutMs?', '+fmt(C.queueTimeoutMs)+'ms timeout':''));
  if(C.maxBodySize)g+=row('Max body',fmt(C.maxBodySize)+' B');
  g+=row('Log format',C.logFormat);
  if(C.logLevel==='debug')g+=row('Log level','<span class="warn">debug (verbose)</span>');
  if(C.systemPrompt)g+=row('System prompt',\`<span title="\${C.systemPrompt}" style="cursor:help">set ℹ</span>\`);
  if(C.ollamaOptions)g+=row('Ollama options',\`<span title="\${C.ollamaOptions}" style="cursor:help;font-size:11px">\${C.ollamaOptions.length>40?C.ollamaOptions.slice(0,40)+'…':C.ollamaOptions}</span>\`);
  g+='<div class="sep"></div>';
  g+='<div class="row" style="gap:8px"><a href="/health">health</a><a href="/metrics">metrics JSON</a><a href="/metrics/prometheus">prometheus</a><a href="/v1/models">models</a></div>';
  g+='</div>';

  // ── Requests card ─────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Requests</h2>';
  if(m){
    const routes=Object.entries(m.requests_total||{});
    if(routes.length){routes.forEach(([k,v])=>g+=row(k,fmt(v)));}
    else g+='<div class="row"><span class="lbl" style="color:#8b949e">No requests yet</span></div>';
    g+=row('Errors (5xx)',m.errors_total?m.errors_total:'0',m.errors_total>0?'err':'ok');
  }
  g+='</div>';

  // ── Latency card ──────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Latency</h2>';
  if(m){
    g+=row('p50',ms(m.latency_p50_ms));
    g+=row('p95',ms(m.latency_p95_ms));
    g+=row('p99',ms(m.latency_p99_ms));
    g+=row('Min',ms(m.latency_min_ms));
    g+=row('Max',ms(m.latency_max_ms));
    g+=row('Avg',ms(m.latency_avg_ms));
  }else{g+='<div class="row"><span class="lbl" style="color:#8b949e">No data yet</span></div>';}
  g+='</div>';

  // ── Tokens card ───────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Tokens (this session)</h2>';
  if(m){
    const tot=(m.tokens_input_total||0)+(m.tokens_output_total||0);
    g+=row('Input',fmt(m.tokens_input_total));
    g+=row('Output',fmt(m.tokens_output_total));
    g+=row('Total',fmt(tot));
    if(C.rateLimitRpm){const pct=Math.round((m.tokens_input_total||0)/C.rateLimitRpm*100);g+=row('vs rate limit',pct+'%');}
  }else{g+='<div class="row"><span class="lbl" style="color:#8b949e">No data yet</span></div>';}
  g+='</div>';

  // ── HTTP status codes card ────────────────────────────────────────────────────
  const codes=m&&m.status_codes?Object.entries(m.status_codes):[];
  if(codes.length){
    g+='<div class="card"><h2>HTTP Status Codes</h2>';
    codes.sort().forEach(([code,cnt])=>{
      const cls=code.startsWith('5')?'err':code.startsWith('4')?'warn':'ok';
      g+=row(code,fmt(cnt),cls);
    });
    g+='</div>';
  }

  // ── Per-model usage card ──────────────────────────────────────────────────────
  const models=m&&m.models_usage?Object.entries(m.models_usage):[];
  if(models.length){
    g+='<div class="card"><h2>Model Usage</h2>';
    models.forEach(([model,v],i)=>{
      if(i>0)g+='<div class="sep"></div>';
      g+='<div style="color:#e6edf3;font-size:12px;font-family:monospace;padding:4px 0">'+model+'</div>';
      g+=row('Requests',fmt(v.requests));
      g+=row('Tokens in',fmt(v.tokens_in));
      g+=row('Tokens out',fmt(v.tokens_out));
    });
    g+='</div>';
  }

  document.getElementById('grid').innerHTML=g;
}
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>`;

  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(html);
}

// Accepts OpenAI-format POST /v1/chat/completions requests and forwards them to Ollama,
// applying auth, rate-limiting, timeout, retry, abort, and metrics — same as the Anthropic path.
// Useful for OpenAI-compatible clients (Cursor, Continue, LiteLLM, etc.) without format translation.
async function handleOpenAIChat(req, res) {
  const ollamaBase = getOllamaHost();
  const ac = new AbortController();

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
  let openaiReq;
  try { openaiReq = JSON.parse(body); }
  catch {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (!Array.isArray(openaiReq.messages)) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`messages` is required and must be an array' } }));
    return;
  }

  const effectiveModel = resolveModel(openaiReq.model);
  openaiReq.model = effectiveModel;
  const chatMaxResult = resolveMaxTokens(openaiReq.max_tokens || null);
  if (chatMaxResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: chatMaxResult.error } }));
    return;
  }
  openaiReq.max_tokens = chatMaxResult.value;
  // OLLAMA_OPTIONS: fill in deployment-level params not already in the client request.
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS)) {
    if (!(k in openaiReq)) openaiReq[k] = v;
  }
  // Dedicated env vars take highest precedence (unconditional overwrite).
  if (OLLAMA_NUM_CTX)         openaiReq.num_ctx      = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE)      openaiReq.keep_alive   = OLLAMA_KEEP_ALIVE;
  const streaming = openaiReq.stream === true;
  if (streaming) openaiReq.stream_options = { include_usage: true };

  // Inject PROXY_SYSTEM_PROMPT so operator-level instructions apply to all callers,
  // not just the Anthropic path. Prepended before any existing system message.
  if (PROXY_SYSTEM_PROMPT) {
    const sysIdx = openaiReq.messages.findIndex(m => m.role === 'system');
    if (sysIdx >= 0) {
      const orig = typeof openaiReq.messages[sysIdx].content === 'string'
        ? openaiReq.messages[sysIdx].content : '';
      openaiReq.messages[sysIdx] = {
        ...openaiReq.messages[sysIdx],
        content: orig ? `${PROXY_SYSTEM_PROMPT}\n\n${orig}` : PROXY_SYSTEM_PROMPT,
      };
    } else {
      openaiReq.messages.unshift({ role: 'system', content: PROXY_SYSTEM_PROMPT });
    }
  }

  debugLog(`→ Ollama [${ollamaBase}] (OpenAI passthrough)`, openaiReq);

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(openaiReq),
      signal: ac.signal,
    });
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    if (e.name === 'AbortError') {
      if (timedOut) {
        if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
        if (!res.writableEnded) res.end(JSON.stringify({
          error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms`, code: 'timeout' }
        }));
      } else {
        res.end();
      }
      return;
    }
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'connection_error', message: e.message + (isConnRefused ? ' — is Ollama running? Try: ollama serve' : '') } }));
    return;
  }

  if (!ollamaRes.ok) {
    req.socket.off('close', onClientClose);
    clearTO();
    const err = await ollamaRes.text();
    res.writeHead(ollamaRes.status, { 'Content-Type': 'application/json' });
    res.end(err);
    return;
  }

  // Non-streaming: proxy the JSON response directly.
  if (!streaming) {
    let data;
    try { data = await ollamaRes.json(); }
    catch (e) {
      req.socket.off('close', onClientClose);
      clearTO();
      if (e.name === 'AbortError') {
        if (timedOut) {
          if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
          if (!res.writableEnded) res.end(JSON.stringify({ error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` } }));
        } else {
          res.end();
        }
        return;
      }
      throw e;
    }
    req.socket.off('close', onClientClose);
    clearTO();
    const promptTok     = data.usage?.prompt_tokens     || 0;
    const completionTok = data.usage?.completion_tokens || 0;
    recordTokens(promptTok, completionTok, effectiveModel);
    res._logMeta = { model: effectiveModel, tokensIn: promptTok, tokensOut: completionTok };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
    return;
  }

  // Streaming: pipe SSE lines through verbatim, intercepting the usage chunk for metrics.
  // Splitting on '\n' and re-emitting 'line\n' correctly preserves SSE '\n\n' event separators
  // because blank separator lines become '\n' tokens that write as the required double-newline.
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });
  _metrics.activeStreams++;

  let inputTokens = 0;
  let outputTokens = 0;
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
      buf = lines.pop(); // hold incomplete trailing line

      for (const line of lines) {
        if (line.startsWith('data: ') && line.slice(6).trim() !== '[DONE]') {
          try {
            const parsed = JSON.parse(line.slice(6).trim());
            if (parsed.usage) {
              inputTokens  = parsed.usage.prompt_tokens     || inputTokens;
              outputTokens = parsed.usage.completion_tokens || outputTokens;
            }
          } catch {}
        }
        res.write(line + '\n');
      }
    }
    if (buf) res.write(buf); // flush any trailing partial line
  } catch (e) {
    if (e.name === 'AbortError') {
      if (timedOut && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` } })}\n\n`);
      }
    } else {
      console.error('OpenAI stream error:', e.message);
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { type: 'stream_error', message: e.message } })}\n\n`);
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

// POST /v1/completions — legacy OpenAI text completions API.
// Converts the `prompt` field into a single-turn chat message and forwards to Ollama,
// then converts the chat response back into the text completions envelope that older
// clients (some LiteLLM configs, older Continue builds, scripts) expect.
// Supports streaming (SSE) and non-streaming. Applies full proxy infrastructure:
// auth, rate-limiting, timeout, retry, client-abort, keepalive, CORS, metrics.
async function handleOpenAICompletions(req, res) {
  const ollamaBase = getOllamaHost();
  const ac = new AbortController();

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
  let completionReq;
  try { completionReq = JSON.parse(body); }
  catch {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Request body is not valid JSON' } }));
    return;
  }

  if (completionReq.prompt == null) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`prompt` is required' } }));
    return;
  }

  const effectiveModel = resolveModel(completionReq.model);
  const streaming = completionReq.stream === true;

  const compMaxResult = resolveMaxTokens(completionReq.max_tokens ?? null);
  if (compMaxResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: compMaxResult.error } }));
    return;
  }

  // Normalize prompt to a single string; OpenAI spec allows string or array of strings.
  const promptText = Array.isArray(completionReq.prompt)
    ? completionReq.prompt.join('\n')
    : String(completionReq.prompt);

  // Build an OpenAI chat request so we can forward to Ollama's /v1/chat/completions.
  const chatReq = {
    model: effectiveModel,
    messages: [{ role: 'user', content: promptText }],
    max_tokens: compMaxResult.value,
    stream: streaming,
    ...(streaming && { stream_options: { include_usage: true } }),
  };
  if (completionReq.temperature !== undefined) chatReq.temperature = completionReq.temperature;
  if (completionReq.top_p      !== undefined) chatReq.top_p      = completionReq.top_p;
  if (completionReq.stop       !== undefined) chatReq.stop       = completionReq.stop;
  if (completionReq.seed       !== undefined) chatReq.seed       = completionReq.seed;
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS)) {
    if (!(k in chatReq)) chatReq[k] = v;
  }
  if (OLLAMA_NUM_CTX)    chatReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) chatReq.keep_alive = OLLAMA_KEEP_ALIVE;

  if (PROXY_SYSTEM_PROMPT) {
    chatReq.messages.unshift({ role: 'system', content: PROXY_SYSTEM_PROMPT });
  }

  debugLog(`→ Ollama [${ollamaBase}] (completions)`, chatReq);

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chatReq),
      signal: ac.signal,
    });
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    if (e.name === 'AbortError') {
      if (timedOut) {
        if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
        if (!res.writableEnded) res.end(JSON.stringify({
          error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms`, code: 'timeout' }
        }));
      } else {
        res.end();
      }
      return;
    }
    const isConnRefused = e.cause?.code === 'ECONNREFUSED' || e.message.includes('ECONNREFUSED');
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'connection_error', message: e.message + (isConnRefused ? ' — is Ollama running? Try: ollama serve' : '') } }));
    return;
  }

  if (!ollamaRes.ok) {
    req.socket.off('close', onClientClose);
    clearTO();
    const err = await ollamaRes.text();
    res.writeHead(ollamaRes.status, { 'Content-Type': 'application/json' });
    res.end(err);
    return;
  }

  const completionId = () => 'cmpl_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

  // ── Non-streaming ────────────────────────────────────────────────────────────
  if (!streaming) {
    let data;
    try { data = await ollamaRes.json(); }
    catch (e) {
      req.socket.off('close', onClientClose);
      clearTO();
      if (e.name === 'AbortError') {
        if (timedOut) {
          if (!res.headersSent) res.writeHead(504, { 'Content-Type': 'application/json' });
          if (!res.writableEnded) res.end(JSON.stringify({ error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` } }));
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
    const promptTok     = data.usage?.prompt_tokens     || 0;
    const completionTok = data.usage?.completion_tokens || 0;
    recordTokens(promptTok, completionTok, effectiveModel);
    res._logMeta = { model: effectiveModel, tokensIn: promptTok, tokensOut: completionTok };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      id: completionId(),
      object: 'text_completion',
      created: Math.floor(Date.now() / 1000),
      model: effectiveModel,
      choices: [{
        text: choice?.message?.content || '',
        index: 0,
        logprobs: null,
        finish_reason: choice?.finish_reason || 'stop',
      }],
      usage: {
        prompt_tokens:     promptTok,
        completion_tokens: completionTok,
        total_tokens:      promptTok + completionTok,
      },
    }));
    return;
  }

  // ── Streaming ─────────────────────────────────────────────────────────────────
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });
  _metrics.activeStreams++;

  const id = completionId();
  const created = Math.floor(Date.now() / 1000);
  let inputTokens = 0;
  let outputTokens = 0;
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
        if (!line.startsWith('data: ')) { res.write(line + '\n'); continue; }
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') { res.write('data: [DONE]\n\n'); continue; }
        let chunk;
        try { chunk = JSON.parse(raw); } catch { res.write(line + '\n'); continue; }

        if (chunk.usage) {
          inputTokens  = chunk.usage.prompt_tokens     || inputTokens;
          outputTokens = chunk.usage.completion_tokens || outputTokens;
        }

        const choice = chunk.choices?.[0];
        if (!choice) continue;

        // Convert chat delta to completions delta.
        const text = choice.delta?.content || '';
        const completionsChunk = {
          id, object: 'text_completion', created, model: effectiveModel,
          choices: [{ text, index: 0, logprobs: null, finish_reason: choice.finish_reason || null }],
        };
        res.write(`data: ${JSON.stringify(completionsChunk)}\n\n`);
      }
    }
    if (buf) res.write(buf);
  } catch (e) {
    if (e.name === 'AbortError') {
      if (timedOut && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { type: 'timeout_error', message: `Ollama did not respond within ${PROXY_TIMEOUT}ms` } })}\n\n`);
      }
    } else {
      console.error('Completions stream error:', e.message);
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { type: 'stream_error', message: e.message } })}\n\n`);
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
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleMessages(req, res);
    } else if (req.method === 'POST' && path === '/v1/chat/completions') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleOpenAIChat(req, res);
    } else if (req.method === 'POST' && path === '/v1/completions') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleOpenAICompletions(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      await handleCountTokens(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/batches') {
      if (!checkAuth(req, res)) return;
      await handleCreateBatch(req, res);
    } else if (req.method === 'GET' && path === '/v1/messages/batches') {
      if (!checkAuth(req, res)) return;
      await handleListBatches(req, res);
    } else if (req.method === 'GET' && path.startsWith('/v1/messages/batches/') && path.endsWith('/results')) {
      if (!checkAuth(req, res)) return;
      await handleGetBatchResults(req, res, path.slice('/v1/messages/batches/'.length, -'/results'.length));
    } else if (req.method === 'POST' && path.startsWith('/v1/messages/batches/') && path.endsWith('/cancel')) {
      if (!checkAuth(req, res)) return;
      await handleCancelBatch(req, res, path.slice('/v1/messages/batches/'.length, -'/cancel'.length));
    } else if (req.method === 'GET' && path.startsWith('/v1/messages/batches/')) {
      if (!checkAuth(req, res)) return;
      await handleGetBatch(req, res, path.slice('/v1/messages/batches/'.length));
    } else if (req.method === 'POST' && path === '/v1/embeddings') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      await handleEmbeddings(req, res);
    } else if (req.method === 'GET' && path === '/v1/models') {
      if (!checkAuth(req, res)) return;
      await handleModels(req, res);
    } else if (req.method === 'GET' && path.startsWith('/v1/models/')) {
      if (!checkAuth(req, res)) return;
      await handleModelById(req, res, decodeURIComponent(path.slice('/v1/models/'.length)));
    } else if (req.method === 'DELETE' && path.startsWith('/v1/models/')) {
      if (!checkAuth(req, res)) return;
      await handleDeleteModel(req, res, decodeURIComponent(path.slice('/v1/models/'.length)));
    } else if (req.method === 'POST' && path === '/v1/models/pull') {
      if (!checkAuth(req, res)) return;
      await handlePullModel(req, res);
    } else if (req.method === 'GET' && (path === '/' || path === '')) {
      handleDashboard(req, res);
    } else if (req.method === 'GET' && path === '/favicon.ico') {
      // Return 204 so browser console stays clean when the dashboard is open.
      res.writeHead(204);
      res.end();
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
    console.log(`  MaxTok: default max_tokens=${PROXY_MAX_TOKENS}${PROXY_HARD_MAX_TOKENS ? ` (hard cap: ${PROXY_HARD_MAX_TOKENS})` : ' (set PROXY_HARD_MAX_TOKENS to cap)'}`);
    console.log(`  MaxBody: ${PROXY_MAX_BODY_SIZE ? `${PROXY_MAX_BODY_SIZE} B per request` : 'unlimited (set PROXY_MAX_BODY_SIZE to limit)'}`);
    if (PROXY_SYSTEM_PROMPT) console.log(`  SysPrompt: ${PROXY_SYSTEM_PROMPT.slice(0, 80)}${PROXY_SYSTEM_PROMPT.length > 80 ? '…' : ''}`);
    console.log(`  Logs  : format=${LOG_FORMAT} level=${LOG_LEVEL} (LOG_FORMAT=json for structured; LOG_LEVEL=debug for full request/response bodies)`);
    console.log(`  Warmup: ${PROXY_WARMUP ? 'enabled — pre-loading model on startup' : 'disabled (set PROXY_WARMUP=true to pre-load model)'}`);
    const rlGlobal = RATE_LIMIT_RPM        ? `global ${RATE_LIMIT_RPM} req/min`    : 'no global limit';
    const rlIp     = RATE_LIMIT_PER_IP_RPM ? `per-IP ${RATE_LIMIT_PER_IP_RPM} req/min` : 'no per-IP limit';
    console.log(`  RateLimit: ${rlGlobal}; ${rlIp} (set RATE_LIMIT_RPM / RATE_LIMIT_PER_IP_RPM)`);
    console.log(`  Concurrency: ${PROXY_MAX_CONCURRENCY ? `max ${PROXY_MAX_CONCURRENCY} simultaneous LLM requests (503 when exceeded)` : 'unlimited (set PROXY_MAX_CONCURRENCY to prevent GPU OOM)'}`);
    if (PROXY_MAX_QUEUE_SIZE)
      console.log(`  Queue    : up to ${PROXY_MAX_QUEUE_SIZE} requests queued${PROXY_MAX_QUEUE_TIMEOUT ? ` (${PROXY_MAX_QUEUE_TIMEOUT}ms timeout)` : ' (no timeout)'}`);
    if (Object.keys(OLLAMA_OPTIONS).length > 0)
      console.log(`  Options: OLLAMA_OPTIONS=${JSON.stringify(OLLAMA_OPTIONS)}`);
    console.log('');

    if (PROXY_WARMUP) {
      // Fire warmup asynchronously so the server is already accepting connections while we load.
      // Uses setImmediate to yield back to the event loop first.
      // Warms all configured hosts in parallel so every GPU is ready, not just the first.
      setImmediate(async () => {
        console.log(`  Warmup: sending preflight to ${OLLAMA_HOSTS.length} host(s) to load ${MODEL} into GPU memory…`);
        await Promise.all(OLLAMA_HOSTS.map(async (ollamaBase) => {
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
              console.log(`  Warmup: ${ollamaBase} ready — model loaded\n`);
            } else {
              const text = await r.text().catch(() => '');
              console.warn(`  Warmup: ${ollamaBase} returned HTTP ${r.status}: ${text.slice(0, 120)}\n`);
            }
          } catch (e) {
            console.warn(`  Warmup: ${ollamaBase} failed (${e.message}) — model will load on first request\n`);
          }
        }));
      });
    }
  });

  // Periodically remove rate-limit windows that have expired (older than one 60 s window).
  // Without cleanup, per-IP windows accumulate indefinitely in long-running deployments
  // where many unique IPs visit once and never return (dynamic IPs, crawlers, etc.).
  setInterval(() => {
    const now = Date.now();
    for (const [key, w] of _rateLimitWindows) {
      if (now - w.windowStart >= 60_000) _rateLimitWindows.delete(key);
    }
  }, 5 * 60_000).unref();

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
  parseDotEnv,
  parseOllamaOptions,
  parseOllamaError,
  OLLAMA_OPTIONS,
  resolveModel,
  resolveMaxTokens,
  PROXY_HARD_MAX_TOKENS,
  toOpenAIMessages,
  toOpenAITools,
  toOpenAIToolChoice,
  extractThinkingParts,
  documentBlockToText,
  imageBlockToOpenAI,
  injectSystemPrompt,
  logRequest,
  sanitizeForLog,
  getOllamaHost,
  OLLAMA_HOSTS,
  requestHandler,
  handleMessages,
  handleOpenAIChat,
  handleOpenAICompletions,
  handleEmbeddings,
  handleDeleteModel,
  handlePullModel,
  handleCreateBatch,
  handleListBatches,
  handleGetBatch,
  handleGetBatchResults,
  handleCancelBatch,
  processBatch,
  processBatchRequest,
  batchRequestCounts,
  batchToResponse,
  _batches,
  handleMetricsPrometheus,
  // Rate-limit internals exported for unit testing only.
  checkRateLimit,
  getClientIp,
  _rateLimitWindows,
  // Concurrency-limit internals exported for unit testing only.
  checkConcurrency,
  acquireLlmSlot,
  releaseLlmSlot,
  trackActiveLlmRequest,
  _concurrencyQueue,
  _metrics,
};
