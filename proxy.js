const http = require('http');

const OLLAMA_BASE = process.env.OLLAMA_HOST || 'http://localhost:11434';
const MODEL      = process.env.OLLAMA_MODEL  || 'qwen2.5:7b';
const PORT       = process.env.PROXY_PORT    || 4000;
const PROXY_API_KEY = process.env.PROXY_API_KEY || null;

// Anthropic messages/tools → OpenAI format
function toOpenAIMessages(messages, system) {
  const result = [];

  const systemText = Array.isArray(system)
    ? system.map(b => b.text || '').join('\n')
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
          ? tr.content.map(c => c.text || '').join('')
          : (tr.content || '');
        const content = tr.is_error ? `[ERROR] ${rawContent}` : rawContent;
        result.push({ role: 'tool', tool_call_id: tr.tool_use_id, content });
      }
      if (textParts.length > 0) {
        result.push({ role: msg.role, content: textParts.map(b => b.text).join('') });
      }
    } else {
      const text = textParts.map(b => b.text).join('');
      const imageParts = blocks.filter(b => b.type === 'image');

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
// Does NOT retry 4xx (client errors) or once the streaming body has begun.
async function fetchWithRetry(url, options, maxRetries = 3, baseDelay = 500) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, options);
      if (res.status < 500 || attempt >= maxRetries) return res;
      console.warn(`Ollama ${res.status}, retrying (${attempt + 1}/${maxRetries})…`);
    } catch (e) {
      if (attempt >= maxRetries) throw e;
      console.warn(`Ollama fetch error: ${e.message}, retrying (${attempt + 1}/${maxRetries})…`);
    }
    await new Promise(r => setTimeout(r, baseDelay * 2 ** attempt));
  }
}

async function handleMessages(req, res) {
  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch { res.writeHead(400); res.end('{"error":"bad json"}'); return; }

  const streaming = anthropicReq.stream !== false;

  // Use request model if it looks like an Ollama model name (not a claude-* alias).
  // This lets callers switch models per-request without restarting the proxy.
  const effectiveModel = (anthropicReq.model && !anthropicReq.model.startsWith('claude-'))
    ? anthropicReq.model
    : MODEL;

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
  if (anthropicReq.stop_sequences?.length) openaiReq.stop = anthropicReq.stop_sequences;

  let ollamaRes;
  try {
    ollamaRes = await fetchWithRetry(`${OLLAMA_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(openaiReq),
      // No hard timeout on the request itself — models can be slow.
      // Connection refused / DNS failures are caught below.
    });
  } catch (e) {
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

  // ── Non-streaming ─────────────────────────────────────────────────────────
  if (!streaming) {
    const data = await ollamaRes.json();
    const choice = data.choices[0];
    const msg = choice.message;

    const content = [];
    if (msg.content) content.push({ type: 'text', text: msg.content });
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
        output_tokens: data.usage?.completion_tokens || 0
      }
    }));
    return;
  }

  // ── Streaming ─────────────────────────────────────────────────────────────
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
      usage: { input_tokens: 0, output_tokens: 0 }
    }
  });

  let textBlockOpen = false;
  const toolBlocks = {}; // openai tool index → { anthropicIndex, id, name, args }
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = null; // set on finish_reason; message_delta deferred until after loop

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

        // Text delta
        if (delta.content) {
          if (!textBlockOpen) {
            sendSSE(res, 'content_block_start', {
              type: 'content_block_start', index: 0,
              content_block: { type: 'text', text: '' }
            });
            sendSSE(res, 'ping', { type: 'ping' });
            textBlockOpen = true;
          }
          sendSSE(res, 'content_block_delta', {
            type: 'content_block_delta', index: 0,
            delta: { type: 'text_delta', text: delta.content }
          });
        }

        // Tool call deltas
        if (delta.tool_calls) {
          for (const tc of delta.tool_calls) {
            const oi = tc.index; // openai index
            if (!toolBlocks[oi]) {
              const ai = (textBlockOpen ? 1 : 0) + oi; // anthropic block index
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

          if (textBlockOpen) {
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: 0 });
          }
          for (const tb of Object.values(toolBlocks)) {
            sendSSE(res, 'content_block_stop', {
              type: 'content_block_stop', index: tb.anthropicIndex
            });
          }
        }
      }
    }
    // All chunks consumed — now emit terminal events with correct token counts.
    if (stopReason && !res.writableEnded) {
      sendSSE(res, 'message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens }
      });
      sendSSE(res, 'message_stop', { type: 'message_stop' });
    }
  } catch (e) {
    console.error('Stream error:', e.message);
    if (!res.writableEnded) {
      sendSSE(res, 'error', { type: 'error', error: { type: 'stream_error', message: e.message } });
    }
  } finally {
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

const server = http.createServer(async (req, res) => {
  const start = Date.now();
  res.on('finish', () => {
    console.log(`${req.method} ${req.url} ${res.statusCode} ${Date.now() - start}ms`);
  });

  if (req.method === 'POST' && req.url === '/v1/messages') {
    if (!checkAuth(req, res)) return;
    await handleMessages(req, res);
  } else if (req.method === 'GET' && req.url === '/v1/models') {
    if (!checkAuth(req, res)) return;
    await handleModels(req, res);
  } else if (req.method === 'GET' && req.url === '/health') {
    await handleHealth(req, res);
  } else {
    res.writeHead(404);
    res.end('{"error":"not found"}');
  }
});

server.listen(PORT, () => {
  console.log(`\n  Claude-Ollama proxy ready`);
  console.log(`  Model : ${MODEL}`);
  console.log(`  Port  : ${PORT}`);
  console.log(`  Ollama: ${OLLAMA_BASE}`);
  console.log(`  Auth  : ${PROXY_API_KEY ? 'enabled (PROXY_API_KEY set)' : 'disabled (open access)'}`);
  console.log(`  Logs  : requests logged to stdout\n`);
});

function shutdown() {
  server.close(() => process.exit(0));
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
