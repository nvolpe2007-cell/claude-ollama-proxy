const http   = require('http');
const https  = require('https');
const fs     = require('fs');
const path   = require('path');
const crypto = require('crypto');

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
// spread load across multiple GPUs or Ollama instances. Trailing slashes are
// stripped — every call site builds URLs as `${host}/api/...`, so a host of
// "http://localhost:11434/" would otherwise produce a double-slash path
// (e.g. "http://localhost:11434//api/tags") that many HTTP servers either
// 404 on or redirect in a way that breaks POST requests.
const OLLAMA_HOSTS = (process.env.OLLAMA_HOST || 'http://localhost:11434')
  .split(',').map(h => h.trim().replace(/\/+$/, '')).filter(Boolean);

// Tracks per-host health so getOllamaHost() can route around a host that is
// down (crashed Ollama instance, unplugged GPU box, etc.) instead of sending
// every Nth request into retries that are guaranteed to fail. A host is
// marked unhealthy after HOST_UNHEALTHY_THRESHOLD consecutive failed checks
// and becomes eligible again as soon as a single check succeeds.
const HOST_UNHEALTHY_THRESHOLD = 2;
const _hostHealth = new Map(OLLAMA_HOSTS.map(h =>
  [h, { healthy: true, consecutiveFailures: 0, lastError: null, lastCheckedAt: null }]));

function recordHostHealth(host, ok, error) {
  const h = _hostHealth.get(host);
  if (!h) return;
  h.lastCheckedAt = new Date().toISOString();
  if (ok) {
    h.healthy = true;
    h.consecutiveFailures = 0;
    h.lastError = null;
  } else {
    h.consecutiveFailures++;
    h.lastError = error;
    if (h.consecutiveFailures >= HOST_UNHEALTHY_THRESHOLD) h.healthy = false;
  }
}

// Pings a single Ollama host's /api/tags endpoint and records the result in
// _hostHealth. Used by the periodic background health checker and by
// GET /health (which performs a live check on every call).
// On success, also stashes the host's available model names in
// _hostHealth[url].models so GET /health can warn when the configured
// model hasn't been pulled yet.
async function checkHostHealth(url) {
  try {
    const r = await fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(3000) });
    recordHostHealth(url, r.ok, r.ok ? null : `HTTP ${r.status}`);
    if (r.ok) {
      try {
        const data = await r.json();
        const h = _hostHealth.get(url);
        if (h) h.models = Array.isArray(data.models) ? data.models.map(m => m.name) : [];
      } catch { /* leave previously known models in place on parse error */ }
    }
    return r.ok;
  } catch (e) {
    recordHostHealth(url, false, e.message);
    return false;
  }
}

// Returns the distinct Ollama model names the proxy may route requests to:
// the default OLLAMA_MODEL plus every MODEL_MAP target. Used by GET /health
// to warn when a configured model hasn't been pulled into Ollama yet.
function getConfiguredModelNames() {
  return [...new Set([MODEL, ...Object.values(MODEL_MAP)])];
}

// Round-robin index. Node.js is single-threaded so no lock is needed.
let _hostIdx = 0;
function getOllamaHost() {
  // Skip hosts currently marked unhealthy. If every host is unhealthy, fail
  // open and return the next host in rotation anyway — existing per-request
  // retry/error handling still applies, and this avoids a total outage just
  // because the health checker hasn't caught up with a recovery yet.
  const start = _hostIdx;
  let fallback = null;
  for (let i = 0; i < OLLAMA_HOSTS.length; i++) {
    const idx = (start + i) % OLLAMA_HOSTS.length;
    const host = OLLAMA_HOSTS[idx];
    if (fallback === null) fallback = host;
    if (_hostHealth.get(host)?.healthy !== false) {
      _hostIdx = (idx + 1) % OLLAMA_HOSTS.length;
      return host;
    }
  }
  _hostIdx = (start + 1) % OLLAMA_HOSTS.length;
  return fallback;
}

const MODEL         = process.env.OLLAMA_MODEL    || 'qwen2.5:7b';
const PORT          = process.env.PROXY_PORT      || 4000;
const PROXY_API_KEY    = process.env.PROXY_API_KEY    || null;
// PROXY_LISTEN_HOST restricts the address the proxy binds to.
// Default (unset) listens on all interfaces (0.0.0.0 / ::).
// Set to '127.0.0.1' to accept connections only from localhost — useful when
// running on a shared or internet-facing machine without a firewall.
const PROXY_LISTEN_HOST = process.env.PROXY_LISTEN_HOST || null;
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
// Idle stream timeout (ms). If set, the proxy aborts a streaming response when no new
// tokens are received from Ollama for this many milliseconds — useful for detecting a
// model that has stalled mid-generation without needing PROXY_TIMEOUT to be set very high.
// Only applies during the streaming phase (after the first SSE event is sent).
// Unset by default (no idle timeout).
const PROXY_IDLE_TIMEOUT = process.env.PROXY_IDLE_TIMEOUT ? Number(process.env.PROXY_IDLE_TIMEOUT) : null;
// When true, unconditionally adds think:true to every outgoing Ollama inference request,
// enabling native chain-of-thought reasoning for thinking models (DeepSeek-R1, Qwen3-thinking,
// etc.) without requiring the client (Claude Code, Cursor, etc.) to send
// thinking:{type:"enabled"} on each request. Safe to set on non-thinking models — Ollama
// ignores the parameter when the model doesn't support it. Per-request client thinking
// parameters still apply on top of this baseline.
const PROXY_FORCE_THINK = process.env.PROXY_FORCE_THINK === 'true';
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
// When true AND OLLAMA_NUM_CTX is set, the proxy automatically drops the oldest
// user/assistant turns from the message history whenever the estimated input token
// count would exceed OLLAMA_NUM_CTX. Prevents "context length exceeded" 400 errors
// from Ollama without requiring callers to manage history length themselves.
// Opt-in (default false) so existing deployments see no behaviour change.
const PROXY_AUTO_TRUNCATE = process.env.PROXY_AUTO_TRUNCATE === 'true';

// Optional path to a JSON file where the Messages Batch API state (requests + results)
// is persisted after every change and reloaded at startup. Without this, batches and
// their results live only in memory and are silently lost if the proxy restarts —
// a real problem for batches that can take hours to process. Opt-in (default null,
// in-memory only) so existing deployments see no behaviour change.
const PROXY_BATCH_PERSIST_PATH = process.env.PROXY_BATCH_PERSIST_PATH || null;

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

// Parses the optional PROXY_API_KEYS env var into a list of { name, key } pairs for
// multi-caller deployments. Format: comma-separated "name:key" entries, e.g.
// "nick:sk-abc123,family:sk-def456,laptop:sk-ghi789". Entries without a colon are
// auto-named key1, key2, ... in declaration order. PROXY_API_KEY (if set) is always
// included first, labeled "default", so existing single-key deployments are unaffected.
// Lets an operator hand out separate keys per device/user, see per-key usage in
// /metrics, and revoke a single key without rotating the shared secret.
// Exported for unit testing.
function parseApiKeys(singleKey, listStr) {
  const keys = [];
  if (singleKey) keys.push({ name: 'default', key: singleKey });
  if (listStr) {
    let n = 0;
    for (const entry of listStr.split(',')) {
      const trimmed = entry.trim();
      if (!trimmed) continue;
      n++;
      const ci = trimmed.indexOf(':');
      if (ci > 0) keys.push({ name: trimmed.slice(0, ci).trim(), key: trimmed.slice(ci + 1).trim() });
      else keys.push({ name: `key${n}`, key: trimmed });
    }
  }
  return keys;
}
const _apiKeys = parseApiKeys(PROXY_API_KEY, process.env.PROXY_API_KEYS);

// Parses the optional PROXY_API_KEY_MODELS env var into a Map of key-name → Set(allowed
// Ollama model names) for multi-caller deployments. Format: comma-separated
// "name:model1|model2|..." entries, e.g. "family:qwen2.5:7b,kids:llama3.2:1b". The name
// matches a PROXY_API_KEYS/PROXY_API_KEY name (or "default" when no named keys are
// configured). Models are pipe-separated since model names themselves contain colons
// (e.g. "qwen2.5:7b"). Keys with no entry here have unrestricted model access — this is
// purely an opt-in allow-list. Lets an operator give e.g. a "family" key access to only a
// small model while "nick" gets the full lineup, without running separate proxy instances.
// Exported for unit testing.
function parseApiKeyModels(str) {
  const map = new Map();
  if (!str) return map;
  for (const entry of str.split(',')) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const ci = trimmed.indexOf(':');
    if (ci <= 0) continue; // malformed entry — skip
    const name = trimmed.slice(0, ci).trim();
    const models = trimmed.slice(ci + 1).split('|').map(m => m.trim()).filter(Boolean);
    if (name && models.length) map.set(name, new Set(models));
  }
  return map;
}
const _apiKeyModels = parseApiKeyModels(process.env.PROXY_API_KEY_MODELS);

// Returns null if the caller (identified by req._apiKeyName, or "default" when no named
// key matched) is allowed to use effectiveModel, or a descriptive error message string
// if not. No restriction applies when the caller's key has no entry in
// PROXY_API_KEY_MODELS — this check is purely additive on top of existing auth.
// Exported for unit testing.
function checkModelAccess(req, effectiveModel) {
  const keyName = req._apiKeyName || 'default';
  const allowed = _apiKeyModels.get(keyName);
  if (!allowed) return null;
  if (allowed.has(effectiveModel)) return null;
  return `API key '${keyName}' is not permitted to use model '${effectiveModel}'. Allowed: ${[...allowed].join(', ')}`;
}

// Returns true if the caller may see/use the Ollama model named realModelName, per
// checkModelAccess/PROXY_API_KEY_MODELS. Used by GET /v1/models and GET /v1/models/:id
// to filter model listings so a restricted key (e.g. a "kids" key limited to one small
// model) only sees models it can actually use — keeping model-picker UIs (Continue,
// Open WebUI, Cursor) from offering choices that would 403 if selected.
// Exported for unit testing.
function isModelVisibleToCaller(req, realModelName) {
  return checkModelAccess(req, realModelName) === null;
}

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

// Maps an Ollama HTTP error response to an Anthropic-style error envelope.
// Ollama 4xx errors (unknown model, bad request, etc.) are surfaced with their
// real status and a matching Anthropic error type instead of a blanket 502, so
// SDK clients — which retry 5xx/overloaded errors — don't burn through retries
// on something a retry can never fix, like a typo'd model name.
function mapOllamaError(status, errText) {
  const message = parseOllamaError(errText);
  switch (status) {
    case 400: return { status: 400, type: 'invalid_request_error', message };
    case 404: return { status: 404, type: 'not_found_error', message };
    case 429: return { status: 429, type: 'rate_limit_error', message };
    default:  return { status: 502, type: 'ollama_error', message };
  }
}

// Logs obj as pretty JSON under a label when LOG_LEVEL=debug. No-op otherwise.
function debugLog(label, obj) {
  if (LOG_LEVEL !== 'debug') return;
  console.log(`[DEBUG] ${label}:\n${JSON.stringify(sanitizeForLog(obj), null, 2)}`);
}

// Optional request rate limits. All apply to POST /v1/messages, /v1/chat/completions,
// /v1/completions, /v1/messages/count_tokens, /v1/embeddings, and /v1/messages/batches
// (batch creation) — the last one matters because a single batch can enqueue many
// inference requests against the same shared Ollama/GPU resource the other limits
// protect, so creation itself must be capped too. Unset (disabled) by default.
// RATE_LIMIT_RPM         — global cap across all callers (requests / minute).
// RATE_LIMIT_PER_IP_RPM  — per-client-IP cap (requests / minute); see PROXY_TRUST_PROXY below
//                          for how the client IP is determined.
// RATE_LIMIT_PER_KEY_RPM — per-API-key cap (requests / minute); buckets by the
//                          caller's matched PROXY_API_KEYS name (or 'default' when
//                          no API keys are configured, or PROXY_API_KEY is used),
//                          so one caller in a multi-key deployment can't exhaust the
//                          shared budget even from behind a NAT shared with other callers.
const RATE_LIMIT_RPM         = process.env.RATE_LIMIT_RPM         ? Number(process.env.RATE_LIMIT_RPM)         : null;
const RATE_LIMIT_PER_IP_RPM  = process.env.RATE_LIMIT_PER_IP_RPM  ? Number(process.env.RATE_LIMIT_PER_IP_RPM)  : null;
const RATE_LIMIT_PER_KEY_RPM = process.env.RATE_LIMIT_PER_KEY_RPM ? Number(process.env.RATE_LIMIT_PER_KEY_RPM) : null;

// By default the proxy does NOT trust the x-forwarded-for header when computing a
// client's IP for RATE_LIMIT_PER_IP_RPM, because PROXY_LISTEN_HOST defaults to all
// interfaces (0.0.0.0) — many deployments are reached directly, not through a reverse
// proxy, and a client-supplied x-forwarded-for header is otherwise trivial to spoof to
// get a fresh rate-limit bucket on every request. Set PROXY_TRUST_PROXY=true only when
// the proxy sits behind a reverse proxy/load balancer that overwrites x-forwarded-for
// with the real client IP before it reaches Node.
const PROXY_TRUST_PROXY = process.env.PROXY_TRUST_PROXY === 'true';

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
  apiKeysUsed:  {},   // 'key-name' → { requests, tokensIn, tokensOut }
};

// Queue of { onGranted } entries waiting for a concurrency slot.
const _concurrencyQueue = [];

// recordRequest runs on every request via the `finish` listener registered before
// checkAuth() (see requestHandler), so `path` is raw, unauthenticated, attacker-controlled
// req.url. Without a cap, a caller with no API key at all could flood unique paths
// (`/a`, `/b`, `/c`, ...) and grow `_metrics.requests` without bound for the life of the
// process — the same unbounded-growth bug class as MAX_BATCH_REQUESTS above, just on the
// metrics store instead of the batch store. Capping distinct keys and bucketing the rest
// under a per-method overflow key keeps memory and /metrics response size bounded; the
// real route table is a small fixed set so this never triggers in normal operation.
const MAX_METRICS_PATH_KEYS = 200;

function recordRequest(method, path, status, ms) {
  const k = `${method} ${path}`;
  if (_metrics.requests[k] !== undefined) {
    _metrics.requests[k]++;
  } else if (Object.keys(_metrics.requests).length < MAX_METRICS_PATH_KEYS) {
    _metrics.requests[k] = 1;
  } else {
    const overflowKey = `${method} (other)`;
    _metrics.requests[overflowKey] = (_metrics.requests[overflowKey] || 0) + 1;
  }
  const s = String(status);
  _metrics.statusCodes[s] = (_metrics.statusCodes[s] || 0) + 1;
  if (status >= 500) _metrics.errors++;
  if (_metrics.latencies.length >= 1000) _metrics.latencies.shift();
  _metrics.latencies.push(ms);
}

function recordTokens(input, output, model, apiKeyName) {
  _metrics.tokensIn  += input;
  _metrics.tokensOut += output;
  if (model) {
    if (!_metrics.modelsUsed[model])
      _metrics.modelsUsed[model] = { requests: 0, tokensIn: 0, tokensOut: 0 };
    _metrics.modelsUsed[model].requests  += 1;
    _metrics.modelsUsed[model].tokensIn  += input;
    _metrics.modelsUsed[model].tokensOut += output;
  }
  if (apiKeyName) {
    if (!_metrics.apiKeysUsed[apiKeyName])
      _metrics.apiKeysUsed[apiKeyName] = { requests: 0, tokensIn: 0, tokensOut: 0 };
    _metrics.apiKeysUsed[apiKeyName].requests  += 1;
    _metrics.apiKeysUsed[apiKeyName].tokensIn  += input;
    _metrics.apiKeysUsed[apiKeyName].tokensOut += output;
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
  if (!requestedModel || typeof requestedModel !== 'string') return MODEL;
  if (MODEL_MAP[requestedModel]) return MODEL_MAP[requestedModel];
  if (requestedModel.startsWith('claude-')) {
    for (const [key, target] of Object.entries(MODEL_MAP)) {
      if (requestedModel.startsWith(key)) return target;
    }
    return MODEL;
  }
  return requestedModel;
}

// Validates that a request's `model` field, if present, is a string — as required
// by both the Anthropic and OpenAI APIs. Returns { error } when invalid, or {}
// when the field is absent/null or a valid string. Catches malformed values
// (numbers, booleans, arrays, objects) with a clear 400 invalid_request_error
// instead of a confusing 500 from resolveModel()'s String.prototype.startsWith()
// call. Exported for unit testing.
function validateModelField(model) {
  if (model !== undefined && model !== null && typeof model !== 'string') {
    return { error: '`model` must be a string' };
  }
  return {};
}

// Validates POST /v1/embeddings' optional `encoding_format` field. The real OpenAI API
// accepts only "float" (default) or "base64"; several popular clients (e.g. LangChain's
// OpenAIEmbeddings) request "base64" explicitly and decode the response as a base64-encoded
// float32 buffer, so silently ignoring this field and always returning a plain float array
// would hand those clients a string they then try to base64-decode as numbers — garbage in,
// no error. Returns { error } when invalid, or {} when absent/null or one of the two values.
// Exported for unit testing.
function validateEncodingFormat(format) {
  if (format === undefined || format === null) return {};
  if (format !== 'float' && format !== 'base64') {
    return { error: '`encoding_format` must be "float" or "base64"' };
  }
  return {};
}

// Encodes an embedding vector the way the real OpenAI API does for encoding_format:"base64" —
// a base64 string of the IEEE-754 float32 values in little-endian byte order.
function embeddingToBase64(embedding) {
  return Buffer.from(Float32Array.from(embedding).buffer).toString('base64');
}

// Validates a request's optional `tools` field. toOpenAITools() calls `.map()` on this
// value and reads `.name` off each entry, so a non-array value (string, object) or a
// malformed entry (null, missing/non-string name) would otherwise throw a TypeError that
// surfaces as an opaque 500 internal_error instead of a clear 400. Returns { error } when
// invalid, or {} when tools is absent/null or a well-formed array. Exported for unit testing.
function validateTools(tools) {
  if (tools === undefined || tools === null) return {};
  if (!Array.isArray(tools)) return { error: '`tools` must be an array' };
  for (const t of tools) {
    if (!t || typeof t !== 'object' || Array.isArray(t) || typeof t.name !== 'string' || !t.name) {
      return { error: 'each item in `tools` must be an object with a non-empty string `name`' };
    }
  }
  return {};
}

// Validates a request's optional `system` field. Per the Anthropic API spec, `system`
// must be a string or an array of content blocks. injectSystemPrompt() spreads non-string
// values (`...system`) when PROXY_SYSTEM_PROMPT is set, so a non-array, non-string,
// truthy value (number, boolean, object) would throw a TypeError that surfaces as an
// opaque 500 internal_error instead of a clear 400. toOpenAIMessages() also reads
// `.type`/`.text` off each array element unguarded, so a malformed element (e.g. `null`)
// needs the same per-element check already applied to `messages` content blocks.
// Returns { error } when invalid, or {} when system is absent/null or a valid string/array
// of well-formed blocks. Exported for unit testing.
function validateSystemField(system) {
  if (system === undefined || system === null) return {};
  if (typeof system === 'string') return {};
  if (Array.isArray(system)) {
    for (const block of system) {
      if (!block || typeof block !== 'object' || Array.isArray(block) || typeof block.type !== 'string') {
        return { error: 'each item in `system` must be an object with a string `type`' };
      }
    }
    return {};
  }
  return { error: '`system` must be a string or an array of content blocks' };
}

// Validates a request's `messages` array (already confirmed to be an array by the
// caller). toOpenAIMessages() and the OpenAI-passthrough handlers read `.role`,
// `.content`, and `.type` off each message/content-block without guarding against
// null or non-object entries, so e.g. `messages: [null]` or a content block of
// `null` would throw a TypeError that surfaces as an opaque 500 internal_error
// instead of a clear 400. Each message must be a non-null object with a string
// `role`; an optional `content` must be a string or an array of non-null objects
// with a string `type` (covers both Anthropic content blocks and OpenAI content
// parts); `tool_result` blocks' own `content` array is checked the same way.
// Returns { error } when invalid, or {} when messages is well-formed.
// Exported for unit testing.
function validateMessages(messages) {
  const isBlock = (b) => b && typeof b === 'object' && !Array.isArray(b) && typeof b.type === 'string';
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object' || Array.isArray(msg)) {
      return { error: 'each item in `messages` must be an object' };
    }
    if (typeof msg.role !== 'string') {
      return { error: 'each item in `messages` must have a string `role`' };
    }
    const { content } = msg;
    if (content === undefined || content === null || typeof content === 'string') continue;
    if (!Array.isArray(content)) {
      return { error: '`content` must be a string or an array of content blocks' };
    }
    for (const block of content) {
      if (!isBlock(block)) {
        return { error: 'each content block must be an object with a string `type`' };
      }
      if (block.type === 'tool_result' && Array.isArray(block.content)) {
        for (const c of block.content) {
          if (!isBlock(c)) {
            return { error: 'each `tool_result` content block must be an object with a string `type`' };
          }
        }
      }
    }
  }
  return {};
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

// Trims a messages array to stay within maxInputTokens (estimated via chars/4).
// Strategy: preserves the system message, always keeps at least the last KEEP_LAST
// non-system messages, and removes complete turns from the oldest part of history.
// After each removal it skips past any orphaned tool/assistant messages at the new
// head so the remaining history always starts with a user role (valid OpenAI format).
// Returns { messages, droppedCount } — exported for unit testing.
function truncateToContext(messages, maxInputTokens) {
  const estimate = (arr) => Math.ceil(JSON.stringify(arr).length / 4);
  if (estimate(messages) <= maxInputTokens) return { messages, droppedCount: 0 };

  const sysMsg    = messages.find(m => m.role === 'system') || null;
  let   history   = messages.filter(m => m.role !== 'system');
  const KEEP_LAST = 2;
  let   droppedCount = 0;

  while (history.length > KEEP_LAST) {
    const current = sysMsg ? [sysMsg, ...history] : history;
    if (estimate(current) <= maxInputTokens) break;
    history.shift();
    droppedCount++;
    // Skip orphaned non-user messages at the head (maintains valid OpenAI format).
    while (history.length > KEEP_LAST && history[0]?.role !== 'user') {
      history.shift();
      droppedCount++;
    }
  }

  // Post-loop cleanup: if the outer while stopped at KEEP_LAST but the head is still
  // not a user message (e.g. last 2 are [assistant, user]), drop orphaned non-user
  // messages until we reach a user or only 1 message remains.
  while (history.length > 1 && history[0]?.role !== 'user') {
    history.shift();
    droppedCount++;
  }

  return { messages: sysMsg ? [sysMsg, ...history] : history, droppedCount };
}

// Content block types toOpenAIMessages() knows how to convert. Anything else (e.g.
// `redacted_thinking`, `server_tool_use`, `web_search_tool_result`, or any block type
// added to the Anthropic API after this proxy's release) falls back to its `text` field
// when present, and is otherwise dropped — either way a [content-block] warning is logged
// so operators can see that a round-tripped conversation lost fidelity instead of it
// happening silently.
const KNOWN_CONTENT_BLOCK_TYPES = new Set(['text', 'image', 'tool_use', 'tool_result', 'thinking', 'document']);

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
    for (const b of blocks) {
      if (!KNOWN_CONTENT_BLOCK_TYPES.has(b.type)) {
        const hasTextFallback = typeof b.text === 'string';
        console.warn(`[content-block] dropping unsupported content block type '${b.type}' from a ${msg.role} message${hasTextFallback ? ' (kept its text field)' : ''}`);
      }
    }
    const toolResults = blocks.filter(b => b.type === 'tool_result');
    const toolUses = blocks.filter(b => b.type === 'tool_use');
    // Unknown block types with a string `text` field (a common forward-compat shape)
    // fall back to that text instead of vanishing entirely.
    const textParts = blocks.filter(b => b.type === 'text' || (!KNOWN_CONTENT_BLOCK_TYPES.has(b.type) && typeof b.text === 'string'));

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
        // All image blocks had an unsupported/missing source (e.g. Files API
        // references) — fall back to a plain string so Ollama doesn't receive
        // a message with an empty content array.
        if (content.length === 0) content = '';
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
// Constant-time string comparison so an invalid API key can't be brute-forced
// by measuring response-time differences. crypto.timingSafeEqual throws if the
// two buffers differ in length, so a same-length dummy comparison is performed
// first to keep the running time independent of the candidate's length too.
function timingSafeEqual(a, b) {
  const aBuf = Buffer.from(a);
  const bBuf = Buffer.from(b);
  if (aBuf.length !== bBuf.length) {
    crypto.timingSafeEqual(aBuf, aBuf);
    return false;
  }
  return crypto.timingSafeEqual(aBuf, bBuf);
}

// Checks the request's key against every configured key (PROXY_API_KEY +
// PROXY_API_KEYS). Iterates the full list rather than returning early so the
// response time doesn't leak which entry (if any) matched. On success, stashes
// the matched key's name on req._apiKeyName for per-key usage tracking.
function checkAuth(req, res) {
  if (_apiKeys.length === 0) return true;
  const fromHeader = req.headers['x-api-key']
    || (req.headers['authorization'] || '').replace(/^Bearer\s+/i, '').trim();
  let matchedName = null;
  for (const { name, key } of _apiKeys) {
    if (timingSafeEqual(fromHeader, key)) matchedName = name;
  }
  if (matchedName) {
    req._apiKeyName = matchedName;
    return true;
  }
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

// Returns the client IP. Only consults x-forwarded-for when PROXY_TRUST_PROXY=true
// (i.e. the operator has confirmed a trusted reverse proxy sets this header) — otherwise
// a direct caller could spoof a fresh IP on every request to dodge RATE_LIMIT_PER_IP_RPM.
function getClientIp(req) {
  if (PROXY_TRUST_PROXY) {
    const xff = req.headers['x-forwarded-for'];
    if (xff) return xff.split(',')[0].trim();
  }
  return req.socket.remoteAddress || 'unknown';
}

// Builds the RATE_LIMIT_PER_KEY_RPM bucket key from the caller's matched API key name
// (set by checkAuth on req._apiKeyName). Callers without a matched key — including all
// callers when no PROXY_API_KEY(S) are configured — share a single 'key:default' bucket.
function rateLimitKeyForRequest(req) {
  return `key:${req._apiKeyName || 'default'}`;
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

// Batch-specific concurrency slot acquisition. Unlike acquireLlmSlot, there is no HTTP
// response to send a 503 to, so this simply waits indefinitely until a slot is available.
// Increments activeLlmRequests when granted; the caller must call releaseLlmSlot() when done.
// This ensures batch items respect PROXY_MAX_CONCURRENCY and compete fairly with real-time
// requests rather than bypassing the gate and potentially causing GPU VRAM OOM errors.
async function acquireLlmSlotForBatch() {
  if (!PROXY_MAX_CONCURRENCY) {
    _metrics.activeLlmRequests++;
    return;
  }
  if (_metrics.activeLlmRequests < PROXY_MAX_CONCURRENCY) {
    _metrics.activeLlmRequests++;
    return;
  }
  // All concurrency slots are taken — queue and wait. Incrementing queuedLlmRequests here
  // ensures that /metrics and the dashboard accurately reflect batch items waiting for a slot.
  _metrics.queuedLlmRequests++;
  await new Promise(resolve => {
    _concurrencyQueue.push({
      onGranted: () => {
        _metrics.queuedLlmRequests--;
        resolve();
      },
    });
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

  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    throw e;
  }
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

  const messagesResult = validateMessages(anthropicReq.messages);
  if (messagesResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: messagesResult.error } }));
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

  const modelResult = validateModelField(anthropicReq.model);
  if (modelResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: modelResult.error } }));
    return;
  }

  const toolsResult = validateTools(anthropicReq.tools);
  if (toolsResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: toolsResult.error } }));
    return;
  }

  const systemResult = validateSystemField(anthropicReq.system);
  if (systemResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: systemResult.error } }));
    return;
  }

  // Anthropic API spec: stream defaults to false when not specified.
  const streaming = anthropicReq.stream === true;

  // Use request model if it looks like an Ollama model name (not a claude-* alias).
  // This lets callers switch models per-request without restarting the proxy.
  const effectiveModel = resolveModel(anthropicReq.model);

  const accessError = checkModelAccess(req, effectiveModel);
  if (accessError) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: accessError } }));
    return;
  }

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
  if (anthropicReq.thinking?.type === 'enabled' || PROXY_FORCE_THINK) openaiReq.think = true;
  // OLLAMA_OPTIONS: fill in Ollama-specific params not already set by the request.
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS)) {
    if (!(k in openaiReq)) openaiReq[k] = v;
  }
  // Dedicated env vars take highest precedence (unconditional overwrite).
  if (OLLAMA_NUM_CTX)    openaiReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) openaiReq.keep_alive = OLLAMA_KEEP_ALIVE;

  // Auto-truncate message history to prevent "context length exceeded" errors.
  // Only active when PROXY_AUTO_TRUNCATE=true and OLLAMA_NUM_CTX is set.
  if (PROXY_AUTO_TRUNCATE && OLLAMA_NUM_CTX) {
    const origEst = Math.ceil(JSON.stringify(openaiReq.messages).length / 4);
    const { messages: trimmed, droppedCount } = truncateToContext(openaiReq.messages, OLLAMA_NUM_CTX);
    if (droppedCount > 0) {
      openaiReq.messages = trimmed;
      const newEst = Math.ceil(JSON.stringify(trimmed).length / 4);
      console.warn(`[auto-truncate] Dropped ${droppedCount} message(s): ~${origEst} → ~${newEst} est. tokens (OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX})`);
      res.setHeader('x-context-truncated', String(droppedCount));
    }
  }

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
    req.socket.off('close', onClientClose);
    clearTO();
    const errText = await ollamaRes.text().catch(() => '');
    const { status, type, message } = mapOllamaError(ollamaRes.status, errText);
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type, message } }));
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
    if (!choice || !choice.message) {
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
        try { input = JSON.parse(tc.function.arguments); }
        catch { console.warn(`[tool-call] Model returned non-JSON arguments for tool "${tc.function?.name}", defaulting to {}: ${tc.function?.arguments}`); }
        content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
      }
    }

    const promptTok = data.usage?.prompt_tokens || 0;
    const completionTok = data.usage?.completion_tokens || 0;
    recordTokens(promptTok, completionTok, effectiveModel, req._apiKeyName);
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
  res.setHeader('X-Accel-Buffering', 'no');
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });
  _metrics.activeStreams++;

  const id = newMsgId();
  // Estimate input tokens from the serialised messages we're about to send.
  // The accurate count arrives in Ollama's trailing usage chunk and is reported in
  // message_delta, but the Anthropic SDK reads input_tokens from message_start —
  // sending 0 there means all SDK clients report 0 input tokens for the session.
  // chars/4 is the standard rough estimate; it gives a plausible non-zero value
  // with zero added latency since openaiReq is already built.
  const approxInputTokens = Math.ceil(JSON.stringify(openaiReq.messages).length / 4);
  sendSSE(res, 'message_start', {
    type: 'message_start',
    message: {
      id, type: 'message', role: 'assistant', content: [],
      model: effectiveModel, stop_reason: null, stop_sequence: null,
      usage: { input_tokens: approxInputTokens, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }
    }
  });

  let textBlockOpen = false;
  let textBlockIdx  = -1;      // anthropic index of the currently open text block
  const toolBlocks  = {};      // openai tool index → { anthropicIndex, id, name, args }
  let inputTokens   = 0;
  let outputTokens  = 0;
  let stopReason    = null;    // set on finish_reason; message_delta deferred until after loop
  let streamErrored = false;   // set when Ollama sends a mid-stream {"error":...} chunk

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

  // Flushes any pending thinking text and closes every currently-open content block
  // (thinking, text, tool_use). Shared by the normal finish_reason path, the
  // no-finish_reason fallback, and the mid-stream error path below so all three
  // leave the SSE stream in a valid state before the terminal event is sent.
  function closeAllBlocks() {
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

  // Prevent reverse-proxy read timeouts on slow models.
  const keepAlive = setInterval(() => res.writableEnded || res.write(': keepalive\n\n'), 15_000);

  // Idle-stream timeout: abort if Ollama stops sending tokens mid-generation.
  let idleTimerId = null;
  let idleTimedOut = false;
  const clearIdle = () => { if (idleTimerId) { clearTimeout(idleTimerId); idleTimerId = null; } };
  const resetIdle = () => {
    if (!PROXY_IDLE_TIMEOUT) return;
    clearIdle();
    idleTimerId = setTimeout(() => {
      idleTimedOut = true;
      timedOut = true;
      if (!res.writableEnded) ac.abort();
      console.warn(`Stream idle timeout after ${PROXY_IDLE_TIMEOUT}ms — no new tokens from Ollama`);
    }, PROXY_IDLE_TIMEOUT);
  };
  resetIdle();

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetIdle();

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

        // Ollama can send a {"error": ...} chunk mid-stream (e.g. the model crashes
        // or runs out of VRAM after generation has already started). Without this
        // check the chunk has no `choices`, so it was silently skipped and the
        // stream ended as if it had completed normally — stop_reason: 'end_turn' —
        // hiding the failure from the client.
        if (chunk.error) {
          const message = typeof chunk.error === 'string'
            ? chunk.error
            : (chunk.error.message || JSON.stringify(chunk.error));
          closeAllBlocks();
          sendSSE(res, 'error', { type: 'error', error: { type: 'api_error', message } });
          streamErrored = true;
          break;
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
          closeAllBlocks();
        }
      }
      if (streamErrored) break;
    }
    // All chunks consumed — emit terminal events with correct token counts.
    // Skipped when a mid-stream error chunk already closed the blocks and sent an
    // 'error' event above; real Anthropic streams don't send message_stop after an error.
    if (!streamErrored) {
      // Guard against streams that end without an explicit finish_reason.
      if (!stopReason) {
        stopReason = 'end_turn';
        closeAllBlocks();
      }
      if (!res.writableEnded) {
        sendSSE(res, 'message_delta', {
          type: 'message_delta',
          delta: { stop_reason: stopReason, stop_sequence: null },
          usage: { input_tokens: inputTokens, output_tokens: outputTokens, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }
        });
        sendSSE(res, 'message_stop', { type: 'message_stop' });
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      if (timedOut && !res.writableEnded) {
        const timeoutMsg = idleTimedOut
          ? `No tokens received for ${PROXY_IDLE_TIMEOUT}ms — stream appears stuck`
          : `Ollama did not respond within ${PROXY_TIMEOUT}ms`;
        sendSSE(res, 'error', {
          type: 'error',
          error: { type: 'request_timeout', message: timeoutMsg }
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
    clearIdle();
    clearInterval(keepAlive);
    _metrics.activeStreams--;
  }

  recordTokens(inputTokens, outputTokens, effectiveModel, req._apiKeyName);
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

  // Restricted API keys (PROXY_API_KEY_MODELS) only see models they're actually
  // permitted to use — keeps model-picker UIs from listing choices that would 403.
  const visibleOllamaModels = (data.models || []).filter(m => isModelVisibleToCaller(req, m.name));

  const ollamaModels = visibleOllamaModels.map(m => ({
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
    .filter(([, target]) => isModelVisibleToCaller(req, target))
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

  // Restricted API keys (PROXY_API_KEY_MODELS) get the same 404 as a model that
  // doesn't exist — consistent with GET /v1/models already hiding it from the list,
  // and avoids confirming the existence of models the caller can't use.
  if (!isModelVisibleToCaller(req, model.name)) {
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
  const deleteAccessError = checkModelAccess(req, modelId);
  if (deleteAccessError) {
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: deleteAccessError } }));
    return;
  }

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

  const pullAccessError = checkModelAccess(req, pullReq.model);
  if (pullAccessError) {
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: pullAccessError } }));
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
  res.setHeader('X-Accel-Buffering', 'no');
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

// Real Anthropic Message Batches API caps a batch at 100,000 requests. Enforcing the
// same limit here prevents a single POST /v1/messages/batches call from creating an
// unbounded in-memory batch — every request's full messages/tools/system payload is
// retained in `batch.requests` until the batch ends (and longer still if
// PROXY_BATCH_PERSIST_PATH writes it to disk), so an unbounded array is an easy way
// for one caller to exhaust proxy memory/disk with a single request.
const MAX_BATCH_REQUESTS = 100000;

function newBatchId() {
  return 'msgbatch_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// ── Batch persistence (PROXY_BATCH_PERSIST_PATH) ──────────────────────────────
// Serializes _batches (a Map of batches, each holding a Map of results) to JSON.
// Writes are serialized: if a save is already in flight when another is requested,
// a pending flag triggers exactly one more save afterwards so the file always
// converges on the latest state without overlapping writes corrupting it.
let _batchSaveInProgress = false;
let _batchSavePending    = false;

async function saveBatchesToDisk(filePath = PROXY_BATCH_PERSIST_PATH) {
  if (!filePath) return;
  if (_batchSaveInProgress) { _batchSavePending = true; return; }
  _batchSaveInProgress = true;
  try {
    const data = [..._batches.values()].map(b => ({ ...b, results: [...b.results.entries()] }));
    await fs.promises.writeFile(filePath, JSON.stringify(data));
  } catch (e) {
    console.warn(`[batches] Failed to persist batches to ${filePath}: ${e.message}`);
  } finally {
    _batchSaveInProgress = false;
    if (_batchSavePending) {
      _batchSavePending = false;
      await saveBatchesToDisk(filePath);
    }
  }
}

// Loads previously-persisted batches at startup. Batches that had not reached
// 'ended' status when the proxy last stopped are resumed via processBatch, which
// skips any item that already has a result — so partially-completed batches
// continue from where they left off rather than reprocessing finished items.
function loadBatchesFromDisk(filePath = PROXY_BATCH_PERSIST_PATH) {
  if (!filePath) return;
  if (!fs.existsSync(filePath)) return;
  let data;
  try {
    data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    console.warn(`[batches] Failed to load persisted batches from ${filePath}: ${e.message}`);
    return;
  }
  for (const b of data) {
    b.results = new Map(b.results);
    _batches.set(b.id, b);
  }
  console.log(`[batches] Loaded ${data.length} batch(es) from ${filePath}`);
  for (const b of _batches.values()) {
    if (b.status !== 'ended') setImmediate(() => processBatch(b));
  }
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

// Returns the caller's API key name for batch-ownership purposes, defaulting to
// "default" when no named key matched (mirrors checkModelAccess's convention).
function batchOwnerName(req) {
  return req._apiKeyName || 'default';
}

// Returns true if the caller may view/manage this batch. Batches persisted before
// this field existed have no `owner` and are treated as belonging to "default" so
// existing single-key/no-auth deployments keep working unchanged.
// Exported for unit testing.
function batchOwnedByCaller(req, batch) {
  return (batch.owner || 'default') === batchOwnerName(req);
}

// Process a single batch request item — reuses the same conversion logic as
// handleMessages but operates synchronously against Ollama (non-streaming).
// Returns { type: 'succeeded', message } or { type: 'errored', error }.
// apiKeyName attributes the item's token usage to the batch's owning API key in
// _metrics.apiKeysUsed, mirroring every real-time endpoint (handleMessages, etc.).
async function processBatchRequest(anthropicReq, ollamaBase, apiKeyName) {
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
  if (anthropicReq.thinking?.type === 'enabled' || PROXY_FORCE_THINK) openaiReq.think = true;
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS))
    if (!(k in openaiReq)) openaiReq[k] = v;
  if (OLLAMA_NUM_CTX)    openaiReq.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) openaiReq.keep_alive = OLLAMA_KEEP_ALIVE;

  if (PROXY_AUTO_TRUNCATE && OLLAMA_NUM_CTX) {
    const { messages: trimmed, droppedCount } = truncateToContext(openaiReq.messages, OLLAMA_NUM_CTX);
    if (droppedCount > 0) {
      openaiReq.messages = trimmed;
      console.warn(`[auto-truncate/batch] Dropped ${droppedCount} message(s) to fit OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX}`);
    }
  }

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
    const { type, message } = mapOllamaError(ollamaRes.status, errText);
    return { type: 'errored', error: { type, message } };
  }

  let data;
  try { data = await ollamaRes.json(); }
  catch { return { type: 'errored', error: { type: 'ollama_error', message: 'Failed to parse Ollama response' } }; }

  const choice = data.choices?.[0];
  if (!choice || !choice.message)
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
      try { input = JSON.parse(tc.function.arguments); }
      catch { console.warn(`[tool-call] Model returned non-JSON arguments for tool "${tc.function?.name}", defaulting to {}: ${tc.function?.arguments}`); }
      content.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
    }
  }

  const promptTok     = data.usage?.prompt_tokens     || 0;
  const completionTok = data.usage?.completion_tokens || 0;
  recordTokens(promptTok, completionTok, effectiveModel, apiKeyName);

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
// Each item acquires a concurrency slot via acquireLlmSlotForBatch before calling
// Ollama, so batch requests respect PROXY_MAX_CONCURRENCY and compete fairly with
// real-time requests rather than bypassing the gate.
async function processBatch(batch) {
  for (const item of batch.requests) {
    // Already processed — happens when a persisted batch is resumed after a restart.
    if (batch.results.has(item.custom_id)) continue;
    if (batch.cancelRequested) {
      batch.results.set(item.custom_id, { type: 'canceled' });
      saveBatchesToDisk();
      continue;
    }
    // Honour the 24-hour expires_at TTL: mark any remaining items as expired if the
    // batch has already outlived its window, rather than continuing to burn GPU time.
    if (Date.now() >= new Date(batch.expires_at).getTime()) {
      batch.results.set(item.custom_id, { type: 'expired' });
      saveBatchesToDisk();
      continue;
    }
    await acquireLlmSlotForBatch();
    // Re-resolve the host for every item rather than once for the whole batch: a batch
    // can run for hours, and pinning it to whichever host was current at the start defeats
    // both multi-host round-robin load distribution and the automatic-failover health
    // tracking (_hostHealth) that every other handler already gets by calling
    // getOllamaHost() fresh per request.
    const ollamaBase = getOllamaHost();
    const result = await processBatchRequest(item.params, ollamaBase, batch.owner || 'default').catch(e => ({
      type: 'errored',
      error: { type: 'internal_error', message: e.message },
    }));
    releaseLlmSlot();
    batch.results.set(item.custom_id, result);
    saveBatchesToDisk();
  }
  batch.status   = 'ended';
  batch.ended_at = new Date().toISOString();
  saveBatchesToDisk();
}

// Enforces batch TTLs and reclaims memory for old ended batches.
// Called on a 5-minute interval when the server is running, and exported for unit tests.
//   • In-progress batches past expires_at → mark unresolved items as {type:'expired'},
//     set status to 'ended'. (Covers batches whose processBatch loop has stalled or not
//     yet started.)
//   • Ended batches whose ended_at is more than 1 hour ago → deleted from the Map so
//     their result data doesn't accumulate indefinitely in long-running deployments.
function cleanupExpiredBatches() {
  const now = Date.now();
  const endedCutoff = new Date(now - 60 * 60 * 1000).toISOString();
  let changed = false;
  for (const [id, batch] of _batches) {
    // Remove batches that ended more than 1 hour ago.
    if (batch.status === 'ended' && batch.ended_at && batch.ended_at < endedCutoff) {
      _batches.delete(id);
      changed = true;
      continue;
    }
    // Force-expire batches still in-progress past their TTL.
    if (batch.status !== 'ended' && now >= new Date(batch.expires_at).getTime()) {
      for (const item of batch.requests) {
        if (!batch.results.has(item.custom_id))
          batch.results.set(item.custom_id, { type: 'expired' });
      }
      batch.status   = 'ended';
      batch.ended_at = new Date().toISOString();
      changed = true;
    }
  }
  if (changed) saveBatchesToDisk();
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

  if (batchReq.requests.length > MAX_BATCH_REQUESTS) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `\`requests\` must contain at most ${MAX_BATCH_REQUESTS} items (got ${batchReq.requests.length})` } }));
    return;
  }

  for (const r of batchReq.requests) {
    if (!r || typeof r !== 'object' || Array.isArray(r)) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: 'Each item in `requests` must be an object' } }));
      return;
    }
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
    const batchMessagesResult = validateMessages(r.params.messages);
    if (batchMessagesResult.error) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}': ${batchMessagesResult.error}` } }));
      return;
    }
    const batchModelResult = validateModelField(r.params.model);
    if (batchModelResult.error) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}': ${batchModelResult.error}` } }));
      return;
    }
    const batchToolsResult = validateTools(r.params.tools);
    if (batchToolsResult.error) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}': ${batchToolsResult.error}` } }));
      return;
    }
    const batchSystemResult = validateSystemField(r.params.system);
    if (batchSystemResult.error) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}': ${batchSystemResult.error}` } }));
      return;
    }
    const batchMaxTokensResult = resolveMaxTokens(r.params.max_tokens);
    if (batchMaxTokensResult.error) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Request '${r.custom_id}': ${batchMaxTokensResult.error}` } }));
      return;
    }
    const batchAccessError = checkModelAccess(req, resolveModel(r.params.model));
    if (batchAccessError) {
      res.writeHead(403, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'permission_error', message: `Request '${r.custom_id}': ${batchAccessError}` } }));
      return;
    }
  }

  const now   = new Date();
  const batch = {
    id:                  newBatchId(),
    status:              'in_progress',
    owner:               batchOwnerName(req),
    created_at:          now.toISOString(),
    expires_at:          new Date(now.getTime() + 24 * 60 * 60 * 1000).toISOString(),
    ended_at:            null,
    cancel_initiated_at: null,
    requests:            batchReq.requests,
    results:             new Map(),
    cancelRequested:     false,
  };

  _batches.set(batch.id, batch);
  saveBatchesToDisk();
  setImmediate(() => processBatch(batch));

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

// Parses `limit`/`before_id`/`after_id` query params for GET /v1/messages/batches,
// matching the cursor-pagination convention used across the Anthropic API.
function parseBatchListParams(req) {
  const params = new URLSearchParams((req.url || '').split('?')[1] || '');
  let limit = 20;
  if (params.has('limit')) {
    limit = Number(params.get('limit'));
    if (!Number.isInteger(limit) || limit < 1 || limit > 1000) {
      return { error: '`limit` must be an integer between 1 and 1000' };
    }
  }
  return { limit, before_id: params.get('before_id') || null, after_id: params.get('after_id') || null };
}

async function handleListBatches(req, res) {
  const baseUrl = getBatchBaseUrl(req);
  const parsed  = parseBatchListParams(req);
  if (parsed.error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: parsed.error } }));
    return;
  }
  const { limit, before_id, after_id } = parsed;
  // Newest first, matching the real API's ordering.
  const all = [..._batches.values()].filter(b => batchOwnedByCaller(req, b)).reverse();

  let startIdx, endExclusive;
  if (before_id) {
    // Unknown cursor (incl. one belonging to another caller's batch, already filtered
    // out of `all`) is treated as "nothing before it" rather than erroring, so a
    // pagination cursor can never reveal whether an id exists in another tenant's list.
    const idx = all.findIndex(b => b.id === before_id);
    endExclusive = idx === -1 ? 0 : idx;
    startIdx     = Math.max(0, endExclusive - limit);
  } else {
    startIdx = after_id ? (() => {
      const idx = all.findIndex(b => b.id === after_id);
      return idx === -1 ? all.length : idx + 1;
    })() : 0;
    endExclusive = Math.min(all.length, startIdx + limit);
  }
  const page     = all.slice(startIdx, endExclusive);
  const hasMore  = before_id ? startIdx > 0 : endExclusive < all.length;

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    data:     page.map(b => batchToResponse(b, baseUrl)),
    has_more: hasMore,
    first_id: page[0]?.id                || null,
    last_id:  page[page.length - 1]?.id  || null,
  }));
}

async function handleGetBatch(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch || !batchOwnedByCaller(req, batch)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Batch '${batchId}' not found` } }));
    return;
  }
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

async function handleGetBatchResults(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch || !batchOwnedByCaller(req, batch)) {
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
  if (!batch || !batchOwnedByCaller(req, batch)) {
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
  saveBatchesToDisk();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(batchToResponse(batch, getBatchBaseUrl(req))));
}

// DELETE /v1/messages/batches/{id} — Anthropic only allows deleting a batch once it has
// finished processing; an in-progress batch must be canceled first (matches the real API).
async function handleDeleteBatch(req, res, batchId) {
  const batch = _batches.get(batchId);
  if (!batch || !batchOwnedByCaller(req, batch)) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'not_found_error', message: `Batch '${batchId}' not found` } }));
    return;
  }
  if (batch.status !== 'ended') {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: `Batch '${batchId}' has not ended yet (status: ${batch.status}); cancel it first` } }));
    return;
  }
  _batches.delete(batchId);
  saveBatchesToDisk();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ id: batchId, type: 'message_batch_deleted' }));
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

  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    throw e;
  }
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

  const embedModelResult = validateModelField(embedReq.model);
  if (embedModelResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: embedModelResult.error } }));
    return;
  }

  const encodingFormatResult = validateEncodingFormat(embedReq.encoding_format);
  if (encodingFormatResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: encodingFormatResult.error } }));
    return;
  }

  const effectiveModel = resolveModel(embedReq.model);

  const embedAccessError = checkModelAccess(req, effectiveModel);
  if (embedAccessError) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: embedAccessError } }));
    return;
  }

  const embedBody = { ...OLLAMA_OPTIONS, model: effectiveModel, input: embedReq.input };
  if (OLLAMA_NUM_CTX)    embedBody.num_ctx    = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE) embedBody.keep_alive = OLLAMA_KEEP_ALIVE;

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${ollamaBase}/api/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(embedBody),
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
    const errText = await ollamaRes.text().catch(() => '');
    const { status, type, message } = mapOllamaError(ollamaRes.status, errText);
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type, message } }));
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

  recordTokens(promptTokens, 0, effectiveModel, req._apiKeyName);
  res._logMeta = { model: effectiveModel, tokensIn: promptTokens, tokensOut: 0 };

  const toEmbeddingValue = embedReq.encoding_format === 'base64' ? embeddingToBase64 : (emb) => emb;

  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    object: 'list',
    data: embeddings.map((emb, i) => ({ object: 'embedding', embedding: toEmbeddingValue(emb), index: i })),
    model: effectiveModel,
    usage: { prompt_tokens: promptTokens, total_tokens: promptTokens },
  }));
}

async function handleCountTokens(req, res) {
  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch { res.writeHead(400); res.end('{"error":"bad json"}'); return; }

  if (!Array.isArray(anthropicReq.messages)) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: '`messages` is required and must be an array' } }));
    return;
  }

  const ctMessagesResult = validateMessages(anthropicReq.messages);
  if (ctMessagesResult.error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: ctMessagesResult.error } }));
    return;
  }

  const ctModelResult = validateModelField(anthropicReq.model);
  if (ctModelResult.error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: ctModelResult.error } }));
    return;
  }

  const ctSystemResult = validateSystemField(anthropicReq.system);
  if (ctSystemResult.error) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: ctSystemResult.error } }));
    return;
  }

  const effectiveModel = resolveModel(anthropicReq.model);

  const ctAccessError = checkModelAccess(req, effectiveModel);
  if (ctAccessError) {
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: ctAccessError } }));
    return;
  }

  const ollamaBase = getOllamaHost();

  // Flatten messages + system to a single string for tokenization.
  // Tool schemas are appended as JSON since they consume context.
  const messages = toOpenAIMessages(anthropicReq.messages, injectSystemPrompt(anthropicReq.system));
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
    const ok = await checkHostHealth(url);
    const h = _hostHealth.get(url);
    return {
      url,
      status: ok ? 'ok' : 'unreachable',
      error: ok ? undefined : h?.lastError,
      // 'active' = eligible for round-robin selection; 'skipped' = currently
      // routed around by getOllamaHost() after repeated failures.
      routing: h?.healthy !== false ? 'active' : 'skipped',
    };
  }));
  const anyOk = hostResults.some(h => h.status === 'ok');
  const allOk = hostResults.every(h => h.status === 'ok');
  // Backward-compat fields derived from the first host (single-host deployments unchanged).
  const first = hostResults[0];

  // Check whether the configured model(s) have actually been pulled into Ollama.
  // Union the model lists from every host that returned one (multi-host
  // deployments may have different models pulled on different GPUs).
  const availableModels = new Set();
  let modelsKnown = false;
  for (const url of OLLAMA_HOSTS) {
    const models = _hostHealth.get(url)?.models;
    if (Array.isArray(models)) {
      modelsKnown = true;
      for (const m of models) availableModels.add(m);
    }
  }
  const modelsStatus = {};
  if (modelsKnown) {
    for (const name of getConfiguredModelNames()) modelsStatus[name] = availableModels.has(name);
  }
  const modelAvailable = modelsKnown ? (modelsStatus[MODEL] ?? false) : null;
  const missingModels = modelsKnown
    ? Object.entries(modelsStatus).filter(([, ok]) => !ok).map(([name]) => name)
    : [];

  const status = !allOk ? 'degraded' : (modelAvailable === false ? 'degraded' : 'ok');

  res.writeHead(anyOk ? 200 : 503, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    status,
    proxy: 'running',
    version: PROXY_VERSION,
    hosts: hostResults,
    ollama: first.status === 'ok' ? 'reachable' : 'unreachable',
    ollamaError: first.error || undefined,
    model: MODEL,
    model_available: modelAvailable,
    ...(modelsKnown && Object.keys(modelsStatus).length > 1 ? { models_status: modelsStatus } : {}),
    ...(missingModels.length > 0 ? {
      warning: `Model${missingModels.length > 1 ? 's' : ''} not found on any reachable Ollama host: ${missingModels.join(', ')} — run 'ollama pull <model>'`
    } : {}),
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
  const apiKeysUsage = {};
  for (const [name, m] of Object.entries(_metrics.apiKeysUsed)) {
    apiKeysUsage[name] = { requests: m.requests, tokens_in: m.tokensIn, tokens_out: m.tokensOut };
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
    api_keys_usage:      apiKeysUsage,
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

  out.push('# HELP proxy_api_key_requests_total Total completed LLM requests per named API key (PROXY_API_KEY/PROXY_API_KEYS)');
  out.push('# TYPE proxy_api_key_requests_total counter');
  for (const [name, m] of Object.entries(_metrics.apiKeysUsed)) {
    out.push(`proxy_api_key_requests_total{key_name="${lv(name)}"} ${m.requests}`);
  }
  out.push('');

  out.push('# HELP proxy_api_key_tokens_total Cumulative LLM tokens per named API key partitioned by direction');
  out.push('# TYPE proxy_api_key_tokens_total counter');
  for (const [name, m] of Object.entries(_metrics.apiKeysUsed)) {
    out.push(`proxy_api_key_tokens_total{key_name="${lv(name)}",direction="input"} ${m.tokensIn}`);
    out.push(`proxy_api_key_tokens_total{key_name="${lv(name)}",direction="output"} ${m.tokensOut}`);
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
    auth:               _apiKeys.length > 0,
    apiKeyNames:        _apiKeys.map(k => k.name),
    apiKeyModels:       Object.fromEntries([..._apiKeyModels].map(([k, v]) => [k, [...v]])),
    tls:                !!TLS_CERT,
    logFormat:          LOG_FORMAT,
    logLevel:           LOG_LEVEL,
    maxTokens:          PROXY_MAX_TOKENS,
    hardMaxTokens:      PROXY_HARD_MAX_TOKENS,
    numCtx:             OLLAMA_NUM_CTX,
    keepAlive:          OLLAMA_KEEP_ALIVE,
    rateLimitRpm:       RATE_LIMIT_RPM,
    rateLimitPerIpRpm:  RATE_LIMIT_PER_IP_RPM,
    rateLimitPerKeyRpm: RATE_LIMIT_PER_KEY_RPM,
    trustProxy:         PROXY_TRUST_PROXY,
    warmup:             PROXY_WARMUP,
    timeout:            PROXY_TIMEOUT,
    idleTimeout:        PROXY_IDLE_TIMEOUT,
    maxBodySize:        PROXY_MAX_BODY_SIZE,
    systemPrompt:       PROXY_SYSTEM_PROMPT ? PROXY_SYSTEM_PROMPT.slice(0, 120) + (PROXY_SYSTEM_PROMPT.length > 120 ? '…' : '') : null,
    ollamaOptions:      Object.keys(OLLAMA_OPTIONS).length > 0 ? JSON.stringify(OLLAMA_OPTIONS) : null,
    maxConcurrency:     PROXY_MAX_CONCURRENCY,
    maxQueueSize:       PROXY_MAX_QUEUE_SIZE,
    queueTimeoutMs:     PROXY_MAX_QUEUE_TIMEOUT,
    forceThink:         PROXY_FORCE_THINK,
    autoTruncate:       PROXY_AUTO_TRUNCATE,
    listenHost:         PROXY_LISTEN_HOST,
    batchPersistPath:   PROXY_BATCH_PERSIST_PATH,
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
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function ms(n){return n==null?'—':n.toLocaleString()+'&thinsp;ms'}
function row(l,v,cls){return'<div class="row"><span class="lbl">'+l+'</span><span class="val '+(cls||'')+'">'+v+'</span></div>'}
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
      let v=badge(ok2,'OK',esc(hh.error||'Err'));
      if(hh.routing==='skipped')v+=' <span class="err" title="repeated failures — skipped by round-robin until it recovers">skipped</span>';
      g+=row(hh.url.replace(/^https?:\\/\\//,''),v);
    });
  } else if(h&&h.ollamaError){
    g+=row('Error','<span class="err">'+esc(h.ollamaError)+'</span>');
  }
  if(h&&h.model_available===false){
    g+=row('Model','<span class="err" title="Run: ollama pull '+C.model+'">'+C.model+' — not pulled ⚠</span>');
  }else if(h&&h.model_available===true){
    g+=row('Model','<span class="ok">'+C.model+'</span>');
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
  if(C.listenHost)g+=row('Bind address',C.listenHost);
  if(C.hosts.length===1)g+=row('Ollama host',C.hosts[0].replace(/^https?:\\/\\//,''));
  else g+=row('Ollama hosts',C.hosts.length+' (round-robin)');
  g+=row('Auth',badge(C.auth,C.apiKeyNames.length>1?'Enabled ('+C.apiKeyNames.length+' keys)':'Enabled','Open — no key'));
  Object.entries(C.apiKeyModels).forEach(([name,models])=>{
    g+=row('Models for '+name,\`<span title="\${models.join(', ')}" style="cursor:help">\${models.length} allowed ℹ</span>\`);
  });
  g+=row('TLS',badge(C.tls,'HTTPS','HTTP'));
  g+=row('Default max_tokens',fmt(C.maxTokens));
  if(C.hardMaxTokens)g+=row('Hard max_tokens cap',fmt(C.hardMaxTokens));
  g+=row('Context (num_ctx)',C.numCtx?fmt(C.numCtx):'model default');
  if(C.keepAlive)g+=row('Keep-alive',C.keepAlive);
  if(C.timeout)g+=row('Timeout',fmt(C.timeout)+' ms');
  if(C.idleTimeout)g+=row('Idle timeout',fmt(C.idleTimeout)+' ms');
  if(C.rateLimitRpm)g+=row('Rate limit (global)',fmt(C.rateLimitRpm)+' req/min');
  if(C.rateLimitPerIpRpm)g+=row('Rate limit (per-IP)',fmt(C.rateLimitPerIpRpm)+' req/min ('+(C.trustProxy?'trusting x-forwarded-for':'socket address')+')');
  if(C.rateLimitPerKeyRpm)g+=row('Rate limit (per-key)',fmt(C.rateLimitPerKeyRpm)+' req/min');
  if(C.maxConcurrency)g+=row('Max concurrency',fmt(C.maxConcurrency)+' req');
  if(C.maxQueueSize)g+=row('Queue depth',fmt(C.maxQueueSize)+' req'+(C.queueTimeoutMs?', '+fmt(C.queueTimeoutMs)+'ms timeout':''));
  if(C.maxBodySize)g+=row('Max body',fmt(C.maxBodySize)+' B');
  g+=row('Log format',C.logFormat);
  if(C.logLevel==='debug')g+=row('Log level','<span class="warn">debug (verbose)</span>');
  if(C.systemPrompt)g+=row('System prompt',\`<span title="\${C.systemPrompt}" style="cursor:help">set ℹ</span>\`);
  if(C.forceThink)g+=row('Force thinking','<span class="ok">Enabled (think:true on all requests)</span>');
  if(C.autoTruncate)g+=row('Auto-truncate','<span class="ok">Enabled'+(C.numCtx?' (limit: '+C.numCtx.toLocaleString()+' tok)':' (set OLLAMA_NUM_CTX)')+'</span>');
  if(C.batchPersistPath)g+=row('Batch persistence',\`<span title="\${C.batchPersistPath}" style="cursor:help">enabled ℹ</span>\`);
  if(C.ollamaOptions)g+=row('Ollama options',\`<span title="\${C.ollamaOptions}" style="cursor:help;font-size:11px">\${C.ollamaOptions.length>40?C.ollamaOptions.slice(0,40)+'…':C.ollamaOptions}</span>\`);
  g+='<div class="sep"></div>';
  g+='<div class="row" style="gap:8px"><a href="/health">health</a><a href="/metrics">metrics JSON</a><a href="/metrics/prometheus">prometheus</a><a href="/v1/models">models</a></div>';
  g+='</div>';

  // ── Requests card ─────────────────────────────────────────────────────────────
  g+='<div class="card"><h2>Requests</h2>';
  if(m){
    const routes=Object.entries(m.requests_total||{});
    if(routes.length){routes.forEach(([k,v])=>g+=row(esc(k),fmt(v)));}
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
      g+='<div style="color:#e6edf3;font-size:12px;font-family:monospace;padding:4px 0">'+esc(model)+'</div>';
      g+=row('Requests',fmt(v.requests));
      g+=row('Tokens in',fmt(v.tokens_in));
      g+=row('Tokens out',fmt(v.tokens_out));
    });
    g+='</div>';
  }

  // ── Per-API-key usage card ──────────────────────────────────────────────────────
  // Only shown for multi-key deployments (PROXY_API_KEYS) — a single key's totals
  // would just duplicate the Tokens card above.
  const apiKeys=m&&m.api_keys_usage?Object.entries(m.api_keys_usage):[];
  if(C.apiKeyNames.length>1&&apiKeys.length){
    g+='<div class="card"><h2>API Key Usage</h2>';
    apiKeys.forEach(([name,v],i)=>{
      if(i>0)g+='<div class="sep"></div>';
      g+='<div style="color:#e6edf3;font-size:12px;font-family:monospace;padding:4px 0">'+esc(name)+'</div>';
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

  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    throw e;
  }
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

  const chatMessagesResult = validateMessages(openaiReq.messages);
  if (chatMessagesResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: chatMessagesResult.error } }));
    return;
  }

  const chatModelResult = validateModelField(openaiReq.model);
  if (chatModelResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: chatModelResult.error } }));
    return;
  }

  const effectiveModel = resolveModel(openaiReq.model);

  const chatAccessError = checkModelAccess(req, effectiveModel);
  if (chatAccessError) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: chatAccessError } }));
    return;
  }

  openaiReq.model = effectiveModel;
  // Accept max_completion_tokens as an alias — newer OpenAI SDK versions (used with o1/o3 models)
  // send this field instead of max_tokens. max_tokens takes precedence when both are present.
  const chatMaxResult = resolveMaxTokens(openaiReq.max_tokens ?? openaiReq.max_completion_tokens ?? null);
  if (chatMaxResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: chatMaxResult.error } }));
    return;
  }
  openaiReq.max_tokens = chatMaxResult.value;
  // Remove max_completion_tokens to avoid forwarding conflicting fields to Ollama.
  delete openaiReq.max_completion_tokens;
  // OLLAMA_OPTIONS: fill in deployment-level params not already in the client request.
  for (const [k, v] of Object.entries(OLLAMA_OPTIONS)) {
    if (!(k in openaiReq)) openaiReq[k] = v;
  }
  // Dedicated env vars take highest precedence (unconditional overwrite).
  if (OLLAMA_NUM_CTX)         openaiReq.num_ctx      = OLLAMA_NUM_CTX;
  if (OLLAMA_KEEP_ALIVE)      openaiReq.keep_alive   = OLLAMA_KEEP_ALIVE;
  if (PROXY_FORCE_THINK && !('think' in openaiReq)) openaiReq.think = true;
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

  if (PROXY_AUTO_TRUNCATE && OLLAMA_NUM_CTX) {
    const origEst = Math.ceil(JSON.stringify(openaiReq.messages).length / 4);
    const { messages: trimmed, droppedCount } = truncateToContext(openaiReq.messages, OLLAMA_NUM_CTX);
    if (droppedCount > 0) {
      openaiReq.messages = trimmed;
      const newEst = Math.ceil(JSON.stringify(trimmed).length / 4);
      console.warn(`[auto-truncate] Dropped ${droppedCount} message(s): ~${origEst} → ~${newEst} est. tokens (OLLAMA_NUM_CTX=${OLLAMA_NUM_CTX})`);
      res.setHeader('x-context-truncated', String(droppedCount));
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
    const err = await ollamaRes.text().catch(() => '');
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
    recordTokens(promptTok, completionTok, effectiveModel, req._apiKeyName);
    res._logMeta = { model: effectiveModel, tokensIn: promptTok, tokensOut: completionTok };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
    return;
  }

  // Streaming: pipe SSE lines through verbatim, intercepting the usage chunk for metrics.
  // Splitting on '\n' and re-emitting 'line\n' correctly preserves SSE '\n\n' event separators
  // because blank separator lines become '\n' tokens that write as the required double-newline.
  res.setHeader('X-Accel-Buffering', 'no');
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });
  _metrics.activeStreams++;

  let inputTokens = 0;
  let outputTokens = 0;
  const keepAlive = setInterval(() => res.writableEnded || res.write(': keepalive\n\n'), 15_000);

  let idleTimerId = null;
  let idleTimedOut = false;
  const clearIdle = () => { if (idleTimerId) { clearTimeout(idleTimerId); idleTimerId = null; } };
  const resetIdle = () => {
    if (!PROXY_IDLE_TIMEOUT) return;
    clearIdle();
    idleTimerId = setTimeout(() => {
      idleTimedOut = true;
      timedOut = true;
      if (!res.writableEnded) ac.abort();
      console.warn(`Stream idle timeout after ${PROXY_IDLE_TIMEOUT}ms — no new tokens from Ollama`);
    }, PROXY_IDLE_TIMEOUT);
  };
  resetIdle();

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetIdle();

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
        const timeoutMsg = idleTimedOut
          ? `No tokens received for ${PROXY_IDLE_TIMEOUT}ms — stream appears stuck`
          : `Ollama did not respond within ${PROXY_TIMEOUT}ms`;
        res.write(`data: ${JSON.stringify({ error: { type: 'timeout_error', message: timeoutMsg } })}\n\n`);
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
    clearIdle();
    clearInterval(keepAlive);
    _metrics.activeStreams--;
  }

  recordTokens(inputTokens, outputTokens, effectiveModel, req._apiKeyName);
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

  let body;
  try {
    body = await readBody(req);
  } catch (e) {
    req.socket.off('close', onClientClose);
    clearTO();
    throw e;
  }
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

  const compModelResult = validateModelField(completionReq.model);
  if (compModelResult.error) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'invalid_request_error', message: compModelResult.error } }));
    return;
  }

  const effectiveModel = resolveModel(completionReq.model);

  const compAccessError = checkModelAccess(req, effectiveModel);
  if (compAccessError) {
    req.socket.off('close', onClientClose);
    clearTO();
    res.writeHead(403, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { type: 'permission_error', message: compAccessError } }));
    return;
  }

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
  if (PROXY_FORCE_THINK) chatReq.think = true;

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
    const err = await ollamaRes.text().catch(() => '');
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
    recordTokens(promptTok, completionTok, effectiveModel, req._apiKeyName);
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
  res.setHeader('X-Accel-Buffering', 'no');
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

  let idleTimerId = null;
  let idleTimedOut = false;
  const clearIdle = () => { if (idleTimerId) { clearTimeout(idleTimerId); idleTimerId = null; } };
  const resetIdle = () => {
    if (!PROXY_IDLE_TIMEOUT) return;
    clearIdle();
    idleTimerId = setTimeout(() => {
      idleTimedOut = true;
      timedOut = true;
      if (!res.writableEnded) ac.abort();
      console.warn(`Stream idle timeout after ${PROXY_IDLE_TIMEOUT}ms — no new tokens from Ollama`);
    }, PROXY_IDLE_TIMEOUT);
  };
  resetIdle();

  const reader = ollamaRes.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      resetIdle();

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

        // Pass a mid-stream {"error":...} chunk straight through (no `choices` to
        // convert) so callers see why generation stopped instead of a silent cutoff.
        if (chunk.error) {
          res.write(`data: ${JSON.stringify({ error: chunk.error })}\n\n`);
          continue;
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
        const timeoutMsg = idleTimedOut
          ? `No tokens received for ${PROXY_IDLE_TIMEOUT}ms — stream appears stuck`
          : `Ollama did not respond within ${PROXY_TIMEOUT}ms`;
        res.write(`data: ${JSON.stringify({ error: { type: 'timeout_error', message: timeoutMsg } })}\n\n`);
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
    clearIdle();
    clearInterval(keepAlive);
    _metrics.activeStreams--;
  }

  recordTokens(inputTokens, outputTokens, effectiveModel, req._apiKeyName);
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

  // anthropic-version mirrors what the real Anthropic API returns on all /v1/messages* routes.
  // Strict SDK clients (some versions of the official TypeScript SDK) validate this header.
  if (path.startsWith('/v1/messages')) {
    res.setHeader('anthropic-version', '2023-06-01');
  }
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
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleMessages(req, res);
    } else if (req.method === 'POST' && path === '/v1/chat/completions') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleOpenAIChat(req, res);
    } else if (req.method === 'POST' && path === '/v1/completions') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      if (!await acquireLlmSlot(req, res)) return;
      trackActiveLlmRequest(res);
      await handleOpenAICompletions(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      await handleCountTokens(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/batches') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
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
    } else if (req.method === 'DELETE' && path.startsWith('/v1/messages/batches/')) {
      if (!checkAuth(req, res)) return;
      await handleDeleteBatch(req, res, path.slice('/v1/messages/batches/'.length));
    } else if (req.method === 'GET' && path.startsWith('/v1/messages/batches/')) {
      if (!checkAuth(req, res)) return;
      await handleGetBatch(req, res, path.slice('/v1/messages/batches/'.length));
    } else if (req.method === 'POST' && path === '/v1/embeddings') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      await handleEmbeddings(req, res);
    } else if (req.method === 'GET' && path === '/v1/models') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      await handleModels(req, res);
    } else if (req.method === 'GET' && path.startsWith('/v1/models/')) {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      await handleModelById(req, res, decodeURIComponent(path.slice('/v1/models/'.length)));
    } else if (req.method === 'DELETE' && path.startsWith('/v1/models/')) {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
      await handleDeleteModel(req, res, decodeURIComponent(path.slice('/v1/models/'.length)));
    } else if (req.method === 'POST' && path === '/v1/models/pull') {
      if (!checkAuth(req, res)) return;
      if (RATE_LIMIT_RPM        && !checkRateLimit('global',        RATE_LIMIT_RPM,        req, res)) return;
      if (RATE_LIMIT_PER_IP_RPM && !checkRateLimit(getClientIp(req), RATE_LIMIT_PER_IP_RPM, req, res)) return;
      if (RATE_LIMIT_PER_KEY_RPM && !checkRateLimit(rateLimitKeyForRequest(req), RATE_LIMIT_PER_KEY_RPM, req, res)) return;
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
  loadBatchesFromDisk();

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

  server.listen(PORT, PROXY_LISTEN_HOST || undefined, () => {
    console.log(`\n  Claude-Ollama proxy ready`);
    console.log(`  Model : ${MODEL}`);
    if (Object.keys(MODEL_MAP).length > 0) {
      for (const [k, v] of Object.entries(MODEL_MAP))
        console.log(`  Map   : ${k} → ${v}`);
    }
    console.log(`  Port  : ${PORT}`);
    console.log(`  Bind  : ${PROXY_LISTEN_HOST || '0.0.0.0 (all interfaces — set PROXY_LISTEN_HOST=127.0.0.1 to restrict)'}`);
    if (OLLAMA_HOSTS.length === 1) {
      console.log(`  Ollama: ${OLLAMA_HOSTS[0]}`);
    } else {
      console.log(`  Ollama: round-robin across ${OLLAMA_HOSTS.length} hosts:`);
      for (const h of OLLAMA_HOSTS) console.log(`    - ${h}`);
    }
    if (_apiKeys.length === 0) {
      console.log(`  Auth  : disabled (open access)`);
    } else {
      console.log(`  Auth  : enabled (${_apiKeys.length} key${_apiKeys.length > 1 ? 's' : ''}: ${_apiKeys.map(k => k.name).join(', ')})`);
    }
    if (_apiKeyModels.size > 0) {
      for (const [name, models] of _apiKeyModels)
        console.log(`  Models: key '${name}' restricted to: ${[...models].join(', ')}`);
    }
    console.log(`  TLS   : ${TLS_CERT ? `enabled (cert: ${TLS_CERT})` : 'disabled (HTTP)'}`);
    console.log(`  CORS  : Access-Control-Allow-Origin: ${CORS_ORIGIN}`);
    console.log(`  Ctx   : ${OLLAMA_NUM_CTX ? `num_ctx=${OLLAMA_NUM_CTX}` : 'model default (set OLLAMA_NUM_CTX to override)'}`);
    if (OLLAMA_KEEP_ALIVE) console.log(`  Keep  : keep_alive=${OLLAMA_KEEP_ALIVE}`);
    console.log(`  Timeout: ${PROXY_TIMEOUT ? `${PROXY_TIMEOUT}ms per request` : 'none (set PROXY_TIMEOUT to limit)'}`);
    console.log(`  IdleTimeout: ${PROXY_IDLE_TIMEOUT ? `${PROXY_IDLE_TIMEOUT}ms idle stream timeout` : 'none (set PROXY_IDLE_TIMEOUT to abort stuck streams)'}`);
    console.log(`  MaxTok: default max_tokens=${PROXY_MAX_TOKENS}${PROXY_HARD_MAX_TOKENS ? ` (hard cap: ${PROXY_HARD_MAX_TOKENS})` : ' (set PROXY_HARD_MAX_TOKENS to cap)'}`);
    console.log(`  MaxBody: ${PROXY_MAX_BODY_SIZE ? `${PROXY_MAX_BODY_SIZE} B per request` : 'unlimited (set PROXY_MAX_BODY_SIZE to limit)'}`);
    if (PROXY_SYSTEM_PROMPT) console.log(`  SysPrompt: ${PROXY_SYSTEM_PROMPT.slice(0, 80)}${PROXY_SYSTEM_PROMPT.length > 80 ? '…' : ''}`);
    console.log(`  Logs  : format=${LOG_FORMAT} level=${LOG_LEVEL} (LOG_FORMAT=json for structured; LOG_LEVEL=debug for full request/response bodies)`);
    console.log(`  Think : ${PROXY_FORCE_THINK ? 'forced (think:true on every request — set PROXY_FORCE_THINK=false to disable)' : 'client-controlled (set PROXY_FORCE_THINK=true to always enable for thinking models)'}`);
    console.log(`  Warmup: ${PROXY_WARMUP ? 'enabled — pre-loading model on startup' : 'disabled (set PROXY_WARMUP=true to pre-load model)'}`);
    if (PROXY_AUTO_TRUNCATE) {
      console.log(`  AutoTruncate: enabled — drops oldest turns when est. input > ${OLLAMA_NUM_CTX ? OLLAMA_NUM_CTX + ' tokens' : 'OLLAMA_NUM_CTX (not set — truncation inactive until OLLAMA_NUM_CTX is configured)'}`);
    }
    console.log(`  Batches: ${PROXY_BATCH_PERSIST_PATH ? `persisted to ${PROXY_BATCH_PERSIST_PATH}` : 'in-memory only (set PROXY_BATCH_PERSIST_PATH to survive restarts)'}`);
    const rlGlobal = RATE_LIMIT_RPM         ? `global ${RATE_LIMIT_RPM} req/min`     : 'no global limit';
    const rlIp     = RATE_LIMIT_PER_IP_RPM  ? `per-IP ${RATE_LIMIT_PER_IP_RPM} req/min (${PROXY_TRUST_PROXY ? 'trusting x-forwarded-for' : 'socket address — set PROXY_TRUST_PROXY=true if behind a reverse proxy'})`  : 'no per-IP limit';
    const rlKey    = RATE_LIMIT_PER_KEY_RPM ? `per-key ${RATE_LIMIT_PER_KEY_RPM} req/min` : 'no per-key limit';
    console.log(`  RateLimit: ${rlGlobal}; ${rlIp}; ${rlKey} (set RATE_LIMIT_RPM / RATE_LIMIT_PER_IP_RPM / RATE_LIMIT_PER_KEY_RPM)`);
    console.log(`  Concurrency: ${PROXY_MAX_CONCURRENCY ? `max ${PROXY_MAX_CONCURRENCY} simultaneous LLM requests (503 when exceeded)` : 'unlimited (set PROXY_MAX_CONCURRENCY to prevent GPU OOM)'}`);
    if (PROXY_MAX_QUEUE_SIZE)
      console.log(`  Queue    : up to ${PROXY_MAX_QUEUE_SIZE} requests queued${PROXY_MAX_QUEUE_TIMEOUT ? ` (${PROXY_MAX_QUEUE_TIMEOUT}ms timeout)` : ' (no timeout)'}`);
    if (Object.keys(OLLAMA_OPTIONS).length > 0)
      console.log(`  Options: OLLAMA_OPTIONS=${JSON.stringify(OLLAMA_OPTIONS)}`);
    console.log('');

    // Non-blocking Ollama connectivity check. Skipped when PROXY_WARMUP=true because the
    // warmup request already verifies reachability as part of model pre-loading. Without
    // this check, operators who skip PROXY_WARMUP wouldn't know Ollama is unreachable until
    // the first actual request fails with an opaque 502.
    if (!PROXY_WARMUP) {
      setImmediate(async () => {
        const results = await Promise.all(OLLAMA_HOSTS.map(async (url) => {
          try {
            const r = await fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(3000) });
            return r.ok ? null : `${url} (HTTP ${r.status})`;
          } catch (e) {
            return `${url} (${e.message})`;
          }
        }));
        const failures = results.filter(Boolean);
        if (failures.length > 0) {
          console.warn(`  Warning: ${failures.length} Ollama host(s) unreachable at startup:`);
          failures.forEach(f => console.warn(`    ${f}`));
          console.warn('  Requests will fail until Ollama is running. Start with: ollama serve\n');
        }
      });
    }

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

  // Periodically enforce batch TTLs and free memory for old ended batches.
  // cleanupExpiredBatches handles both force-expiry of stalled in-progress batches
  // and removal of ended batches older than 1 hour.
  setInterval(cleanupExpiredBatches, 5 * 60_000).unref();

  // In multi-host deployments, periodically probe each Ollama host so
  // getOllamaHost() can route around one that has gone down without waiting
  // for a live request to fail first. Single-host setups have nowhere else
  // to route, so this is skipped to avoid needless background traffic.
  if (OLLAMA_HOSTS.length > 1) {
    OLLAMA_HOSTS.forEach(checkHostHealth); // seed initial state immediately
    setInterval(() => OLLAMA_HOSTS.forEach(checkHostHealth), 15_000).unref();
  }

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
  mapOllamaError,
  OLLAMA_OPTIONS,
  resolveModel,
  resolveMaxTokens,
  validateModelField,
  validateTools,
  validateSystemField,
  validateMessages,
  validateEncodingFormat,
  embeddingToBase64,
  PROXY_HARD_MAX_TOKENS,
  PROXY_IDLE_TIMEOUT,
  PROXY_FORCE_THINK,
  PROXY_TRUST_PROXY,
  PROXY_AUTO_TRUNCATE,
  truncateToContext,
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
  // Host-health internals exported for unit testing only.
  checkHostHealth,
  recordHostHealth,
  _hostHealth,
  HOST_UNHEALTHY_THRESHOLD,
  getConfiguredModelNames,
  requestHandler,
  handleMessages,
  handleOpenAIChat,
  handleOpenAICompletions,
  handleEmbeddings,
  handleCountTokens,
  handleDeleteModel,
  handlePullModel,
  handleCreateBatch,
  handleListBatches,
  parseBatchListParams,
  handleGetBatch,
  handleGetBatchResults,
  handleCancelBatch,
  handleDeleteBatch,
  processBatch,
  cleanupExpiredBatches,
  processBatchRequest,
  batchRequestCounts,
  batchToResponse,
  batchOwnedByCaller,
  batchOwnerName,
  saveBatchesToDisk,
  loadBatchesFromDisk,
  PROXY_BATCH_PERSIST_PATH,
  MAX_BATCH_REQUESTS,
  _batches,
  handleMetricsPrometheus,
  // Rate-limit internals exported for unit testing only.
  checkRateLimit,
  getClientIp,
  rateLimitKeyForRequest,
  _rateLimitWindows,
  // Concurrency-limit internals exported for unit testing only.
  checkConcurrency,
  acquireLlmSlot,
  acquireLlmSlotForBatch,
  releaseLlmSlot,
  trackActiveLlmRequest,
  _concurrencyQueue,
  _metrics,
  recordRequest,
  MAX_METRICS_PATH_KEYS,
  recordTokens,
  // Auth internals exported for unit testing only.
  checkAuth,
  timingSafeEqual,
  parseApiKeys,
  _apiKeys,
  parseApiKeyModels,
  checkModelAccess,
  isModelVisibleToCaller,
  _apiKeyModels,
  handleModels,
  handleModelById,
  handleDashboard,
};
