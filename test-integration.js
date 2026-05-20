'use strict';
/**
 * Integration tests for claude-ollama-proxy.
 *
 * Starts a real proxy server against a mock Ollama HTTP server and exercises
 * the full request/response cycle: health, models, non-streaming messages,
 * streaming SSE, tool calls, count_tokens, CORS, auth, and 404 handling.
 *
 * Run standalone: npm run test:integration
 * (must not share a Node process with test.js — proxy.js env vars are module-level constants)
 */

const OLLAMA_PORT = Number(process.env.TEST_OLLAMA_PORT || 19998);
const PROXY_PORT  = Number(process.env.TEST_PROXY_PORT  || 19999);

// Must be set BEFORE proxy.js is first required — constants are evaluated at load time.
process.env.OLLAMA_HOST  = `http://127.0.0.1:${OLLAMA_PORT}`;
process.env.PROXY_PORT   = String(PROXY_PORT);
delete process.env.PROXY_API_KEY;  // most tests run without auth

const { describe, test, before, after, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const http   = require('http');
const { requestHandler } = require('./proxy');

// ── Mock Ollama server ────────────────────────────────────────────────────────

let mockBehavior = 'default';

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks).toString();
}

const mockOllama = http.createServer(async (req, res) => {
  const path = req.url.split('?')[0];

  if (path === '/api/tags') {
    if (mockBehavior === 'ollama-error') {
      res.writeHead(500); res.end('{}'); return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      models: [{ name: 'qwen2.5:7b', modified_at: '2025-01-01T00:00:00Z' }]
    }));
    return;
  }

  if (path === '/v1/chat/completions') {
    const body = JSON.parse(await readBody(req));

    if (mockBehavior === 'tool-call') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        choices: [{
          message: {
            role: 'assistant', content: null,
            tool_calls: [{
              id: 'call_abc', type: 'function',
              function: { name: 'get_weather', arguments: '{"city":"London"}' }
            }]
          },
          finish_reason: 'tool_calls'
        }],
        usage: { prompt_tokens: 20, completion_tokens: 8 }
      }));
      return;
    }

    if (mockBehavior === 'think-response') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        choices: [{
          message: { role: 'assistant', content: '<think>my reasoning</think>my answer' },
          finish_reason: 'stop'
        }],
        usage: { prompt_tokens: 15, completion_tokens: 10 }
      }));
      return;
    }

    if (mockBehavior === 'ollama-5xx') {
      res.writeHead(503); res.end('{}'); return;
    }

    if (mockBehavior === 'streaming-tool-call') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      // First chunk: role announcement with no content
      res.write('data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}\n\n');
      // Tool call header: id + name arrive together in Ollama's first tool chunk
      res.write('data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n');
      // Arguments streamed incrementally across two chunks to test partial-json assembly
      res.write('data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]},"finish_reason":null}]}\n\n');
      res.write('data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"London\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n');
      res.write('data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":15}}\n\n');
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    if (mockBehavior === 'streaming-think') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write('data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n');
      // Split the opening tag + content across a chunk boundary to exercise the state-machine buffer
      res.write('data: {"choices":[{"index":0,"delta":{"content":"<think>rea"},"finish_reason":null}]}\n\n');
      res.write('data: {"choices":[{"index":0,"delta":{"content":"soning</think>"},"finish_reason":null}]}\n\n');
      res.write('data: {"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n');
      res.write('data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}\n\n');
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    if (body.stream) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write('data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n');
      res.write('data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n');
      res.write('data: {"id":"c1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n');
      res.write('data: {"id":"c1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n');
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        choices: [{ message: { role: 'assistant', content: 'Hello!' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5 }
      }));
    }
    return;
  }

  if (path === '/api/tokenize') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ tokens: [1, 2, 3, 4, 5] }));
    return;
  }

  res.writeHead(404); res.end('{}');
});

// ── Proxy server ──────────────────────────────────────────────────────────────

const proxyServer = http.createServer(requestHandler);

// ── Request helper ────────────────────────────────────────────────────────────

function request(method, path, body, headers = {}) {
  return new Promise((resolve, reject) => {
    const payload = body != null ? JSON.stringify(body) : null;
    const h = { 'Content-Type': 'application/json', ...headers };
    if (payload) h['Content-Length'] = Buffer.byteLength(payload);
    const r = http.request(
      { host: '127.0.0.1', port: PROXY_PORT, method, path, headers: h },
      (res) => {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => resolve({
          status: res.statusCode,
          headers: res.headers,
          body: Buffer.concat(chunks).toString()
        }));
      }
    );
    r.on('error', reject);
    if (payload) r.write(payload);
    r.end();
  });
}

function json(r) { return JSON.parse(r.body); }

// Parse an SSE response body into [{event, data}] objects.
function parseSse(text) {
  const events = [];
  let event = null;
  for (const line of text.split('\n')) {
    if (line.startsWith('event: ')) {
      event = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      try { events.push({ event, data: JSON.parse(line.slice(6)) }); } catch {}
      event = null;
    }
  }
  return events;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

before(async () => {
  await new Promise((resolve, reject) => {
    mockOllama.listen(OLLAMA_PORT, '127.0.0.1', resolve);
    mockOllama.on('error', reject);
  });
  await new Promise((resolve, reject) => {
    proxyServer.listen(PROXY_PORT, '127.0.0.1', resolve);
    proxyServer.on('error', reject);
  });
});

after(() => new Promise(resolve => {
  mockOllama.close(() => proxyServer.close(resolve));
}));

beforeEach(() => { mockBehavior = 'default'; });

// ── Tests: GET /health ────────────────────────────────────────────────────────

describe('GET /health', () => {
  test('200 with status:ok when Ollama is reachable', async () => {
    const r = await request('GET', '/health');
    assert.equal(r.status, 200);
    const b = json(r);
    assert.equal(b.status, 'ok');
    assert.equal(b.proxy, 'running');
    assert.equal(b.ollama, 'reachable');
    assert.ok(typeof b.model === 'string' && b.model.length > 0);
    assert.ok(typeof b.port === 'number');
    assert.ok(typeof b.timestamp === 'string');
  });

  test('503 with status:degraded when Ollama returns an error', async () => {
    mockBehavior = 'ollama-error';
    const r = await request('GET', '/health');
    assert.equal(r.status, 503);
    const b = json(r);
    assert.equal(b.status, 'degraded');
    assert.equal(b.ollama, 'unreachable');
  });
});

// ── Tests: GET /v1/models ─────────────────────────────────────────────────────

describe('GET /v1/models', () => {
  test('200 with Anthropic-style model list from Ollama', async () => {
    const r = await request('GET', '/v1/models');
    assert.equal(r.status, 200);
    const b = json(r);
    assert.equal(b.object, 'list');
    assert.ok(Array.isArray(b.data));
    assert.equal(b.data[0].id, 'qwen2.5:7b');
    assert.equal(b.data[0].object, 'model');
    assert.equal(b.data[0].owned_by, 'ollama');
    assert.ok(typeof b.data[0].created === 'number');
  });
});

// ── Tests: GET /metrics ───────────────────────────────────────────────────────

describe('GET /metrics', () => {
  test('200 with all expected metric fields', async () => {
    const r = await request('GET', '/metrics');
    assert.equal(r.status, 200);
    const b = json(r);
    const required = [
      'uptime_seconds', 'requests_total', 'status_codes',
      'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
      'tokens_input_total', 'tokens_output_total',
      'active_streams', 'errors_total'
    ];
    for (const k of required) {
      assert.ok(k in b, `missing field: ${k}`);
    }
    assert.ok(typeof b.uptime_seconds === 'number');
    assert.ok(typeof b.requests_total === 'object');
    assert.ok(b.active_streams === 0);
  });
});

// ── Tests: POST /v1/messages — non-streaming ──────────────────────────────────

describe('POST /v1/messages (non-streaming)', () => {
  test('returns correct Anthropic message envelope', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      stream: false
    });
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('application/json'));
    const b = json(r);
    assert.equal(b.type, 'message');
    assert.equal(b.role, 'assistant');
    assert.ok(b.id.startsWith('msg_'));
    assert.ok(Array.isArray(b.content));
    assert.equal(b.content[0].type, 'text');
    assert.equal(b.content[0].text, 'Hello!');
    assert.ok(['end_turn', 'max_tokens', 'tool_use'].includes(b.stop_reason));
  });

  test('usage includes required token fields including prompt-caching compat', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 100,
      stream: false
    });
    const b = json(r);
    assert.ok(typeof b.usage.input_tokens === 'number');
    assert.ok(typeof b.usage.output_tokens === 'number');
    assert.equal(b.usage.cache_creation_input_tokens, 0);
    assert.equal(b.usage.cache_read_input_tokens, 0);
  });

  test('thinking model response is split into thinking + text content blocks', async () => {
    mockBehavior = 'think-response';
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Think carefully' }],
      max_tokens: 100,
      stream: false
    });
    assert.equal(r.status, 200);
    const b = json(r);
    const thinking = b.content.find(c => c.type === 'thinking');
    const text     = b.content.find(c => c.type === 'text');
    assert.ok(thinking, 'should have thinking block');
    assert.equal(thinking.thinking, 'my reasoning');
    assert.ok(thinking.signature, 'thinking block should carry a signature');
    assert.ok(text, 'should have text block');
    assert.equal(text.text, 'my answer');
  });

  test('tool_use content block for tool-calling responses', async () => {
    mockBehavior = 'tool-call';
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'What is the weather in London?' }],
      tools: [{
        name: 'get_weather',
        description: 'Get current weather',
        input_schema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] }
      }],
      max_tokens: 100,
      stream: false
    });
    assert.equal(r.status, 200);
    const b = json(r);
    const tu = b.content.find(c => c.type === 'tool_use');
    assert.ok(tu, 'should have tool_use block');
    assert.equal(tu.name, 'get_weather');
    assert.deepEqual(tu.input, { city: 'London' });
    assert.ok(tu.id, 'tool_use block should have an id');
    assert.equal(b.stop_reason, 'tool_use');
  });

  test('400 on malformed JSON body', async () => {
    const r = await new Promise((resolve, reject) => {
      const payload = 'not json';
      const opts = {
        host: '127.0.0.1', port: PROXY_PORT,
        method: 'POST', path: '/v1/messages',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }
      };
      const req = http.request(opts, (res) => {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks).toString() }));
      });
      req.on('error', reject);
      req.write(payload);
      req.end();
    });
    assert.equal(r.status, 400);
  });

  test('502 when Ollama is unreachable', async () => {
    mockBehavior = 'ollama-5xx';
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 10,
      stream: false
    });
    // 5xx from Ollama triggers retries then a 502 from the proxy
    assert.equal(r.status, 502);
  });
});

// ── Tests: POST /v1/messages — streaming ──────────────────────────────────────

describe('POST /v1/messages (streaming)', () => {
  test('returns text/event-stream with correct Anthropic SSE event sequence', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      stream: true
    });
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('text/event-stream'));

    const events = parseSse(r.body);
    const types  = events.map(e => e.event);

    // Required Anthropic SSE event types
    for (const t of ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop']) {
      assert.ok(types.includes(t), `missing event: ${t}`);
    }

    // message_start shape
    const start = events.find(e => e.event === 'message_start');
    assert.equal(start.data.type, 'message_start');
    assert.equal(start.data.message.role, 'assistant');
    assert.ok(start.data.message.id.startsWith('msg_'));
    assert.equal(start.data.message.usage.cache_creation_input_tokens, 0);
    assert.equal(start.data.message.usage.cache_read_input_tokens, 0);

    // correct text was streamed
    const textDeltas = events.filter(e => e.event === 'content_block_delta');
    const text = textDeltas.map(e => e.data.delta?.text ?? '').join('');
    assert.equal(text, 'Hello!');

    // message_delta has stop_reason and prompt-caching compat usage
    const delta = events.find(e => e.event === 'message_delta');
    assert.equal(delta.data.delta.stop_reason, 'end_turn');
    assert.ok(typeof delta.data.usage.output_tokens === 'number');
    assert.equal(delta.data.usage.cache_creation_input_tokens, 0);
    assert.equal(delta.data.usage.cache_read_input_tokens, 0);
  });
});

// ── Tests: POST /v1/messages/count_tokens ────────────────────────────────────

describe('POST /v1/messages/count_tokens', () => {
  test('returns input_tokens using Ollama tokenizer', async () => {
    const r = await request('POST', '/v1/messages/count_tokens', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hello world' }]
    });
    assert.equal(r.status, 200);
    const b = json(r);
    assert.ok(typeof b.input_tokens === 'number', 'input_tokens should be a number');
    assert.ok(b.input_tokens > 0, 'input_tokens should be positive');
    assert.equal(b.input_tokens, 5);  // mock tokenizer returns 5 tokens
  });
});

// ── Tests: CORS ───────────────────────────────────────────────────────────────

describe('CORS', () => {
  test('OPTIONS preflight returns 204 with CORS headers', async () => {
    const r = await request('OPTIONS', '/v1/messages', null, {
      Origin: 'https://example.com',
      'Access-Control-Request-Method': 'POST'
    });
    assert.equal(r.status, 204);
    assert.ok(r.headers['access-control-allow-origin'],  'missing ACAO header');
    assert.ok(r.headers['access-control-allow-methods'], 'missing ACAM header');
    assert.ok(r.headers['access-control-allow-headers'], 'missing ACAH header');
  });

  test('all responses carry Access-Control-Allow-Origin', async () => {
    for (const [method, path, body] of [
      ['GET',  '/health',  null],
      ['GET',  '/v1/models', null],
      ['GET',  '/metrics', null],
    ]) {
      const r = await request(method, path, body);
      assert.ok(r.headers['access-control-allow-origin'],
        `missing ACAO on ${method} ${path}`);
    }
  });
});

// ── Tests: Auth enforcement ───────────────────────────────────────────────────

describe('Auth enforcement', () => {
  test('requests succeed when PROXY_API_KEY is not set', async () => {
    // PROXY_API_KEY was deleted at module load time; auth is disabled.
    const r = await request('GET', '/v1/models');
    assert.notEqual(r.status, 401);
    assert.notEqual(r.status, 403);
  });
});

// ── Tests: GET /v1/models/:modelId ───────────────────────────────────────────

describe('GET /v1/models/:modelId', () => {
  test('200 with model object for an existing model', async () => {
    const r = await request('GET', '/v1/models/qwen2.5:7b');
    assert.equal(r.status, 200);
    const b = json(r);
    assert.equal(b.id, 'qwen2.5:7b');
    assert.equal(b.object, 'model');
    assert.equal(b.owned_by, 'ollama');
    assert.ok(typeof b.created === 'number');
  });

  test('404 for a model not in Ollama', async () => {
    const r = await request('GET', '/v1/models/no-such-model:latest');
    assert.equal(r.status, 404);
    const b = json(r);
    assert.equal(b.error.type, 'not_found_error');
  });

  test('502 when Ollama is unreachable', async () => {
    mockBehavior = 'ollama-error';
    const r = await request('GET', '/v1/models/qwen2.5:7b');
    assert.equal(r.status, 502);
  });
});

// ── Tests: input validation ───────────────────────────────────────────────────

describe('POST /v1/messages — input validation', () => {
  test('400 when messages field is missing', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      max_tokens: 100,
      stream: false
      // no messages field
    });
    assert.equal(r.status, 400);
    const b = json(r);
    assert.equal(b.error.type, 'invalid_request_error');
    assert.ok(b.error.message.includes('messages'));
  });

  test('400 when messages is not an array', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: 'not an array',
      max_tokens: 100,
      stream: false
    });
    assert.equal(r.status, 400);
    const b = json(r);
    assert.equal(b.error.type, 'invalid_request_error');
  });
});

// ── Tests: stream defaults to false ──────────────────────────────────────────

describe('POST /v1/messages — stream default', () => {
  test('omitting stream returns non-streaming JSON response', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 100
      // stream not set — should default to false per Anthropic spec
    });
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('application/json'),
      'expected JSON response when stream is omitted');
    const b = json(r);
    assert.equal(b.type, 'message');
  });
});

// ── Tests: request-id response header ────────────────────────────────────────

describe('Response headers — request-id', () => {
  test('every response carries a unique request-id header', async () => {
    const ids = new Set();
    for (const [method, path] of [['GET', '/health'], ['GET', '/v1/models'], ['GET', '/metrics']]) {
      const r = await request(method, path, null);
      assert.ok(r.headers['request-id'], `missing request-id on ${method} ${path}`);
      assert.ok(r.headers['request-id'].startsWith('req_'),
        `request-id should start with req_, got: ${r.headers['request-id']}`);
      ids.add(r.headers['request-id']);
    }
    assert.equal(ids.size, 3, 'each request should have a distinct request-id');
  });

  test('POST /v1/messages response carries request-id', async () => {
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 10,
      stream: false,
    });
    assert.ok(r.headers['request-id'], 'missing request-id on POST /v1/messages');
    assert.ok(r.headers['request-id'].startsWith('req_'));
  });
});

// ── Tests: GET /metrics/prometheus ───────────────────────────────────────────

describe('GET /metrics/prometheus', () => {
  test('returns 200 with Prometheus text content-type', async () => {
    const r = await request('GET', '/metrics/prometheus');
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('text/plain'), `unexpected content-type: ${r.headers['content-type']}`);
    assert.ok(r.headers['content-type']?.includes('version=0.0.4'));
  });

  test('response contains required HELP and TYPE lines for core metrics', () => {
    return request('GET', '/metrics/prometheus').then(r => {
      const body = r.body;
      for (const name of ['proxy_uptime_seconds', 'proxy_requests_total', 'proxy_http_responses_total',
                          'proxy_request_duration_ms', 'proxy_tokens_total', 'proxy_active_streams', 'proxy_errors_total']) {
        assert.ok(body.includes(`# HELP ${name}`), `missing HELP for ${name}`);
        assert.ok(body.includes(`# TYPE ${name}`), `missing TYPE for ${name}`);
      }
    });
  });

  test('uptime is a non-negative integer', async () => {
    const r = await request('GET', '/metrics/prometheus');
    const match = r.body.match(/^proxy_uptime_seconds (\d+)/m);
    assert.ok(match, 'proxy_uptime_seconds line not found');
    assert.ok(Number(match[1]) >= 0);
  });

  test('summary metrics include quantile 0.5, 0.95, 0.99, _sum, and _count lines', async () => {
    const r = await request('GET', '/metrics/prometheus');
    assert.ok(r.body.includes('proxy_request_duration_ms{quantile="0.5"}'));
    assert.ok(r.body.includes('proxy_request_duration_ms{quantile="0.95"}'));
    assert.ok(r.body.includes('proxy_request_duration_ms{quantile="0.99"}'));
    assert.ok(r.body.includes('proxy_request_duration_ms_sum '));
    assert.ok(r.body.includes('proxy_request_duration_ms_count '));
  });

  test('token counters use direction label', async () => {
    const r = await request('GET', '/metrics/prometheus');
    assert.ok(r.body.includes('proxy_tokens_total{direction="input"}'));
    assert.ok(r.body.includes('proxy_tokens_total{direction="output"}'));
  });

  test('request counts appear in output after making requests', async () => {
    await request('GET', '/health');
    const r = await request('GET', '/metrics/prometheus');
    assert.ok(r.body.includes('proxy_requests_total{method="GET",path="/health"}'));
  });

  test('each line is either blank, a comment, or a valid metric line', async () => {
    const r = await request('GET', '/metrics/prometheus');
    for (const line of r.body.split('\n')) {
      if (!line) continue;
      assert.ok(
        line.startsWith('#') || /^[a-z_]+({[^}]*})? /.test(line),
        `unexpected line format: ${line}`
      );
    }
  });
});

// ── Tests: 404 and unknown routes ─────────────────────────────────────────────

describe('Unknown routes', () => {
  test('GET /unknown returns 404', async () => {
    const r = await request('GET', '/unknown-path');
    assert.equal(r.status, 404);
  });

  test('POST /v1/unknown returns 404', async () => {
    const r = await request('POST', '/v1/unknown', {});
    assert.equal(r.status, 404);
  });
});

// ── Tests: Request logging via metrics ────────────────────────────────────────

describe('Request logging (via metrics)', () => {
  test('each handled request increments request count in metrics', async () => {
    // make a known request then verify it appears in requests_total
    await request('GET', '/health');
    const r = await request('GET', '/metrics');
    const b = json(r);
    assert.ok(b.requests_total['GET /health'] > 0,
      'GET /health should appear in requests_total');
  });
});

// ── Tests: POST /v1/messages — streaming tool calls ───────────────────────────

describe('POST /v1/messages (streaming) — tool calls', () => {
  test('streams tool_use block with correct Anthropic SSE events', async () => {
    mockBehavior = 'streaming-tool-call';
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'What is the weather in London?' }],
      tools: [{
        name: 'get_weather',
        description: 'Get current weather',
        input_schema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] }
      }],
      max_tokens: 100,
      stream: true
    });
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('text/event-stream'));

    const events = parseSse(r.body);
    const types  = events.map(e => e.event);

    // Required envelope events
    for (const t of ['message_start', 'content_block_start', 'content_block_stop', 'message_delta', 'message_stop']) {
      assert.ok(types.includes(t), `missing event: ${t}`);
    }

    // content_block_start should describe a tool_use block with name and id
    const toolStart = events.find(e =>
      e.event === 'content_block_start' && e.data.content_block?.type === 'tool_use'
    );
    assert.ok(toolStart, 'should have tool_use content_block_start');
    assert.equal(toolStart.data.content_block.name, 'get_weather');
    assert.ok(toolStart.data.content_block.id, 'tool_use block should have an id');

    // Arguments streamed as input_json_delta events — join and parse to verify correctness
    const jsonDeltas = events.filter(e =>
      e.event === 'content_block_delta' && e.data.delta?.type === 'input_json_delta'
    );
    assert.ok(jsonDeltas.length > 0, 'should have input_json_delta events');
    const fullArgs = jsonDeltas.map(e => e.data.delta.partial_json).join('');
    assert.deepEqual(JSON.parse(fullArgs), { city: 'London' });

    // stop_reason must be tool_use for tool-calling responses
    const delta = events.find(e => e.event === 'message_delta');
    assert.equal(delta.data.delta.stop_reason, 'tool_use');

    // message_stop must be the final event
    assert.equal(types[types.length - 1], 'message_stop');
  });
});

// ── Tests: POST /v1/messages — streaming thinking blocks ─────────────────────

describe('POST /v1/messages (streaming) — thinking blocks', () => {
  test('routes <think> tag content to thinking block then text block', async () => {
    mockBehavior = 'streaming-think';
    const r = await request('POST', '/v1/messages', {
      model: 'claude-3-haiku',
      messages: [{ role: 'user', content: 'Think carefully' }],
      max_tokens: 100,
      stream: true
    });
    assert.equal(r.status, 200);
    assert.ok(r.headers['content-type']?.includes('text/event-stream'));

    const events = parseSse(r.body);

    // Thinking block must open before text block
    const thinkStart = events.find(e =>
      e.event === 'content_block_start' && e.data.content_block?.type === 'thinking'
    );
    assert.ok(thinkStart, 'should have thinking content_block_start');

    // thinking_delta events should reconstruct the full thinking text
    const thinkDeltas = events.filter(e =>
      e.event === 'content_block_delta' && e.data.delta?.type === 'thinking_delta'
    );
    assert.ok(thinkDeltas.length > 0, 'should have thinking_delta events');
    assert.equal(thinkDeltas.map(e => e.data.delta.thinking).join(''), 'reasoning');

    // signature_delta must appear before the thinking block closes (Anthropic protocol)
    const sigDelta = events.find(e =>
      e.event === 'content_block_delta' && e.data.delta?.type === 'signature_delta'
    );
    assert.ok(sigDelta, 'should have signature_delta before thinking block stop');
    assert.ok(sigDelta.data.delta.signature, 'signature_delta must carry a non-empty signature');

    // Text block for the non-thinking part of the response
    const textStart = events.find(e =>
      e.event === 'content_block_start' && e.data.content_block?.type === 'text'
    );
    assert.ok(textStart, 'should have text content_block_start after thinking block');

    const textDeltas = events.filter(e =>
      e.event === 'content_block_delta' && e.data.delta?.type === 'text_delta'
    );
    assert.equal(textDeltas.map(e => e.data.delta.text).join(''), 'answer');

    // thinking block index must be lower than text block index
    assert.ok(thinkStart.data.index < textStart.data.index,
      'thinking block must precede text block in index ordering');

    // Correct stop_reason for a normal completion
    const msgDelta = events.find(e => e.event === 'message_delta');
    assert.equal(msgDelta.data.delta.stop_reason, 'end_turn');
  });
});
