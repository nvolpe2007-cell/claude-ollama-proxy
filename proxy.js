const http  = require('http');
const https = require('https');
const fs    = require('fs');

const OLLAMA_BASE   = process.env.OLLAMA_HOST     || 'http://localhost:11434';
const MODEL         = process.env.OLLAMA_MODEL    || 'qwen2.5:7b';
const PORT          = process.env.PROXY_PORT      || 4000;
const PROXY_API_KEY = process.env.PROXY_API_KEY   || null;
const TLS_CERT      = process.env.PROXY_TLS_CERT  || null;
const TLS_KEY       = process.env.PROXY_TLS_KEY   || null;
// CORS_ORIGIN controls the Access-Control-Allow-Origin header.
// Set to a specific origin (e.g. "https://my-app.example.com") or leave as "*" for open access.
const CORS_ORIGIN   = process.env.CORS_ORIGIN     || '*';

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
      const rawText = textParts.map(b => b.text).join('');
      const imageParts = blocks.filter(b => b.type === 'image');
      const docParts  = blocks.filter(b => b.type === 'document');
      const docTexts  = docParts.map(documentBlockToText).filter(Boolean);
      const text      = [rawText, ...docTexts].filter(Boolean).join('\n\n');

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
  let body = '';
  for await (const chunk of req) body += chunk;
  return body;
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
    await new Promise(r => setTimeout(r, baseDelay * 2 ** attempt));
  }
}

async function handleMessages(req, res) {
  // Abort Ollama request if the client disconnects — avoids wasting GPU compute.
  const ac = new AbortController();
  const onClientClose = () => { if (!res.writableEnded) ac.abort(); };
  req.socket.once('close', onClientClose);

  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch {
    req.socket.off('close', onClientClose);
    res.writeHead(400); res.end('{"error":"bad json"}'); return;
  }

  const streaming = anthropicReq.stream !== false;

  // Use request model if it looks like an Ollama model name (not a claude-* alias).
  // This lets callers switch models per-request without restarting the proxy.
  const effectiveModel = resolveModel(anthropicReq.model);

  const openaiReq = {
    model: effectiveModel,
    messages: toOpenAIMessages(anthropicReq.messages, anthropicReq.system),
    stream: streaming,
    max_tokens: anthropicReq.max_tokens || 8192,
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

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${OLLAMA_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(openaiReq),
      signal: ac.signal,
      // No hard timeout on the request itself — models can be slow.
      // Connection refused / DNS failures are caught below.
    });
  } catch (e) {
    req.socket.off('close', onClientClose);
    if (e.name === 'AbortError') { res.end(); return; }
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
      if (e.name === 'AbortError') { res.end(); return; }
      throw e;
    }
    req.socket.off('close', onClientClose);
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
    if (e.name !== 'AbortError') {
      console.error('Stream error:', e.message);
      if (!res.writableEnded) {
        sendSSE(res, 'error', { type: 'error', error: { type: 'stream_error', message: e.message } });
      }
    }
  } finally {
    req.socket.off('close', onClientClose);
    clearInterval(keepAlive);
  }

  res.end();
}

async function handleModels(req, res) {
  let data;
  try {
    const r = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(5000) });
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

async function handleCountTokens(req, res) {
  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch { res.writeHead(400); res.end('{"error":"bad json"}'); return; }

  const effectiveModel = resolveModel(anthropicReq.model);

  // Flatten messages + system to a single string for tokenization.
  // Tool schemas are appended as JSON since they consume context.
  const messages = toOpenAIMessages(anthropicReq.messages || [], anthropicReq.system);
  let prompt = messages.map(m => {
    if (typeof m.content === 'string') return m.content;
    if (Array.isArray(m.content)) return m.content.map(p => p.text || '').join('');
    return '';
  }).join('\n');
  if (anthropicReq.tools?.length) prompt += '\n' + JSON.stringify(anthropicReq.tools);

  // Try Ollama's native tokenize endpoint; fall back to char/4 estimate.
  let inputTokens;
  try {
    const r = await fetch(`${OLLAMA_BASE}/api/tokenize`, {
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
  let ollamaOk = false;
  let ollamaError = null;
  try {
    const check = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(3000) });
    ollamaOk = check.ok;
    if (!check.ok) ollamaError = `HTTP ${check.status}`;
  } catch (e) {
    ollamaError = e.message;
  }

  const status = ollamaOk ? 200 : 503;
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    status: ollamaOk ? 'ok' : 'degraded',
    proxy: 'running',
    ollama: ollamaOk ? 'reachable' : 'unreachable',
    ollamaError: ollamaError || undefined,
    model: MODEL,
    port: Number(PORT),
    timestamp: new Date().toISOString()
  }));
}

async function requestHandler(req, res) {
  // CORS headers on every response — must happen before any writeHead call.
  setCORSHeaders(res);

  // Respond to browser preflight checks immediately, before auth or body parsing.
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const start = Date.now();
  res.on('finish', () => {
    console.log(`${req.method} ${req.url} ${res.statusCode} ${Date.now() - start}ms`);
  });

  // Strip query string before routing so ?foo=bar variants still match.
  const path = req.url.split('?')[0];

  try {
    if (req.method === 'POST' && path === '/v1/messages') {
      if (!checkAuth(req, res)) return;
      await handleMessages(req, res);
    } else if (req.method === 'POST' && path === '/v1/messages/count_tokens') {
      if (!checkAuth(req, res)) return;
      await handleCountTokens(req, res);
    } else if (req.method === 'GET' && path === '/v1/models') {
      if (!checkAuth(req, res)) return;
      await handleModels(req, res);
    } else if (req.method === 'GET' && path === '/health') {
      await handleHealth(req, res);
    } else {
      res.writeHead(404);
      res.end('{"error":"not found"}');
    }
  } catch (e) {
    console.error('Unhandled request error:', e);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'internal_error', message: e.message } }));
    } else if (!res.writableEnded) {
      res.end();
    }
  }
}

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
  console.log(`  Ollama: ${OLLAMA_BASE}`);
  console.log(`  Auth  : ${PROXY_API_KEY ? 'enabled (PROXY_API_KEY set)' : 'disabled (open access)'}`);
  console.log(`  TLS   : ${TLS_CERT ? `enabled (cert: ${TLS_CERT})` : 'disabled (HTTP)'}`);
  console.log(`  CORS  : Access-Control-Allow-Origin: ${CORS_ORIGIN}`);
  console.log(`  Logs  : requests logged to stdout\n`);
});

function shutdown() {
  server.close(() => process.exit(0));
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
