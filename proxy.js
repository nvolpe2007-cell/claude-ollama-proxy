const http = require('http');

const OLLAMA_BASE = process.env.OLLAMA_HOST || 'http://localhost:11434';
const MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:7b';
const PORT = process.env.PROXY_PORT || 4000;

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
        const content = Array.isArray(tr.content)
          ? tr.content.map(c => c.text || '').join('')
          : (tr.content || '');
        result.push({ role: 'tool', tool_call_id: tr.tool_use_id, content });
      }
      if (textParts.length > 0) {
        result.push({ role: msg.role, content: textParts.map(b => b.text).join('') });
      }
    } else {
      const text = textParts.map(b => b.text).join('');
      const out = { role: msg.role, content: text || '' };
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

function toOpenAITools(tools) {
  if (!tools || tools.length === 0) return undefined;
  return tools.map(t => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }));
}

function sendSSE(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function newMsgId() {
  return 'msg_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

async function readBody(req) {
  let body = '';
  for await (const chunk of req) body += chunk;
  return body;
}

async function handleMessages(req, res) {
  const body = await readBody(req);
  let anthropicReq;
  try { anthropicReq = JSON.parse(body); }
  catch { res.writeHead(400); res.end('{"error":"bad json"}'); return; }

  const streaming = anthropicReq.stream !== false;

  const openaiReq = {
    model: MODEL,
    messages: toOpenAIMessages(anthropicReq.messages, anthropicReq.system),
    stream: streaming,
    max_tokens: anthropicReq.max_tokens || 8192,
  };

  const tools = toOpenAITools(anthropicReq.tools);
  if (tools) openaiReq.tools = tools;
  if (anthropicReq.temperature !== undefined) openaiReq.temperature = anthropicReq.temperature;

  let ollamaRes;
  try {
    ollamaRes = await fetch(`${OLLAMA_BASE}/v1/chat/completions`, {
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
      model: anthropicReq.model,
      stop_reason: choice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn',
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
      model: anthropicReq.model, stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 }
    }
  });

  let textBlockOpen = false;
  const toolBlocks = {}; // openai tool index → { anthropicIndex, id, name, args }
  let outputTokens = 0;

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

        if (chunk.usage) outputTokens = chunk.usage.completion_tokens || 0;

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

        // Finish
        if (choice.finish_reason) {
          const stopReason = choice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn';

          if (textBlockOpen) {
            sendSSE(res, 'content_block_stop', { type: 'content_block_stop', index: 0 });
          }
          for (const tb of Object.values(toolBlocks)) {
            sendSSE(res, 'content_block_stop', {
              type: 'content_block_stop', index: tb.anthropicIndex
            });
          }

          sendSSE(res, 'message_delta', {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: outputTokens }
          });
          sendSSE(res, 'message_stop', { type: 'message_stop' });
        }
      }
    }
  } catch (e) {
    console.error('Stream error:', e.message);
  }

  res.end();
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
    await handleMessages(req, res);
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
  console.log(`  Logging: requests logged to stdout\n`);
});
