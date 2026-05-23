'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');

const {
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
  checkRateLimit,
  getClientIp,
  _rateLimitWindows,
} = require('./proxy');

// ── resolveModel ──────────────────────────────────────────────────────────────

describe('resolveModel', () => {
  const DEFAULT = process.env.OLLAMA_MODEL || 'qwen2.5:7b';

  test('returns default model for falsy input', () => {
    assert.equal(resolveModel(undefined), DEFAULT);
    assert.equal(resolveModel(null), DEFAULT);
    assert.equal(resolveModel(''), DEFAULT);
  });

  test('passes through non-claude model names unchanged', () => {
    assert.equal(resolveModel('llama3'), 'llama3');
    assert.equal(resolveModel('mistral:7b'), 'mistral:7b');
    assert.equal(resolveModel('gemma3:27b'), 'gemma3:27b');
  });

  test('maps unmapped claude-* names to the default model', () => {
    assert.equal(resolveModel('claude-3-haiku'), DEFAULT);
    assert.equal(resolveModel('claude-3-opus-20240229'), DEFAULT);
    assert.equal(resolveModel('claude-sonnet-4-6'), DEFAULT);
  });
});

// ── injectSystemPrompt ────────────────────────────────────────────────────────
// Note: injectSystemPrompt reads the module-level PROXY_SYSTEM_PROMPT constant.
// When that env var is not set (test environment), the function is a no-op.

describe('injectSystemPrompt', () => {
  test('returns system unchanged when PROXY_SYSTEM_PROMPT is not set', () => {
    // In the test environment PROXY_SYSTEM_PROMPT is unset, so this is a no-op.
    assert.equal(injectSystemPrompt('hello'), 'hello');
    assert.equal(injectSystemPrompt(null), null);
    assert.equal(injectSystemPrompt(undefined), undefined);
    assert.deepEqual(injectSystemPrompt([{ type: 'text', text: 'hi' }]),
      [{ type: 'text', text: 'hi' }]);
  });

  test('pure function signature: string + no env → passthrough', () => {
    const result = injectSystemPrompt('be concise');
    assert.equal(result, 'be concise');
  });

  test('pure function signature: array + no env → passthrough', () => {
    const blocks = [{ type: 'text', text: 'be concise' }];
    assert.deepEqual(injectSystemPrompt(blocks), blocks);
  });
});

// ── extractThinkingParts ──────────────────────────────────────────────────────

describe('extractThinkingParts', () => {
  test('returns null when no <think> tags present', () => {
    assert.equal(extractThinkingParts('Hello world'), null);
    assert.equal(extractThinkingParts(''), null);
    assert.equal(extractThinkingParts('< not a tag >'), null);
  });

  test('extracts a single think block with no surrounding text', () => {
    const parts = extractThinkingParts('<think>inner thought</think>');
    assert.deepEqual(parts, [{ type: 'thinking', thinking: 'inner thought' }]);
  });

  test('extracts think block followed by text', () => {
    const parts = extractThinkingParts('<think>thought</think>answer');
    assert.deepEqual(parts, [
      { type: 'thinking', thinking: 'thought' },
      { type: 'text', text: 'answer' },
    ]);
  });

  test('extracts text followed by think block', () => {
    const parts = extractThinkingParts('preamble<think>thought</think>');
    assert.deepEqual(parts, [
      { type: 'text', text: 'preamble' },
      { type: 'thinking', thinking: 'thought' },
    ]);
  });

  test('handles multiple think blocks with interleaved text', () => {
    const parts = extractThinkingParts('<think>t1</think>mid<think>t2</think>end');
    assert.deepEqual(parts, [
      { type: 'thinking', thinking: 't1' },
      { type: 'text', text: 'mid' },
      { type: 'thinking', thinking: 't2' },
      { type: 'text', text: 'end' },
    ]);
  });

  test('handles multiline content inside think blocks', () => {
    const parts = extractThinkingParts('<think>line1\nline2\nline3</think>done');
    assert.equal(parts[0].thinking, 'line1\nline2\nline3');
    assert.equal(parts[1].text, 'done');
  });

  test('trims surrounding whitespace from text parts', () => {
    const parts = extractThinkingParts('<think>t</think>  answer  ');
    assert.equal(parts.find(p => p.type === 'text')?.text, 'answer');
  });
});

// ── documentBlockToText ───────────────────────────────────────────────────────

describe('documentBlockToText', () => {
  test('returns null when no source', () => {
    assert.equal(documentBlockToText({}), null);
  });

  test('returns text data from text source', () => {
    const block = { source: { type: 'text', data: 'hello doc' } };
    assert.equal(documentBlockToText(block), 'hello doc');
  });

  test('prepends title header when title is set', () => {
    const block = { title: 'My File', source: { type: 'text', data: 'content' } };
    assert.equal(documentBlockToText(block), '[Document: My File]\ncontent');
  });

  test('decodes base64 text/* sources', () => {
    const data = Buffer.from('decoded text').toString('base64');
    const block = { source: { type: 'base64', media_type: 'text/plain', data } };
    assert.equal(documentBlockToText(block), 'decoded text');
  });

  test('decodes base64 text/html source', () => {
    const data = Buffer.from('<b>bold</b>').toString('base64');
    const block = { source: { type: 'base64', media_type: 'text/html', data } };
    assert.equal(documentBlockToText(block), '<b>bold</b>');
  });

  test('returns placeholder for non-text base64 sources', () => {
    const block = { source: { type: 'base64', media_type: 'application/pdf', data: 'abc' } };
    const result = documentBlockToText(block);
    assert.ok(result.includes('application/pdf'));
    assert.ok(result.includes('not supported'));
  });

  test('returns placeholder for URL sources', () => {
    const block = { source: { type: 'url', url: 'https://example.com/doc.pdf' } };
    const result = documentBlockToText(block);
    assert.ok(result.includes('https://example.com/doc.pdf'));
    assert.ok(result.includes('not fetched'));
  });

  test('returns null for unknown source type', () => {
    const block = { source: { type: 'unknown' } };
    assert.equal(documentBlockToText(block), null);
  });
});

// ── imageBlockToOpenAI ────────────────────────────────────────────────────────

describe('imageBlockToOpenAI', () => {
  test('returns null for block with no source', () => {
    assert.equal(imageBlockToOpenAI({}), null);
  });

  test('converts base64 image source to data URL', () => {
    const block = { source: { type: 'base64', media_type: 'image/png', data: 'abc123' } };
    const result = imageBlockToOpenAI(block);
    assert.deepEqual(result, {
      type: 'image_url',
      image_url: { url: 'data:image/png;base64,abc123' },
    });
  });

  test('converts URL image source to image_url', () => {
    const block = { source: { type: 'url', url: 'https://example.com/img.png' } };
    const result = imageBlockToOpenAI(block);
    assert.deepEqual(result, {
      type: 'image_url',
      image_url: { url: 'https://example.com/img.png' },
    });
  });

  test('returns null for unknown source type', () => {
    assert.equal(imageBlockToOpenAI({ source: { type: 'other' } }), null);
  });
});

// ── toOpenAITools ─────────────────────────────────────────────────────────────

describe('toOpenAITools', () => {
  test('returns undefined for null, undefined, or empty array', () => {
    assert.equal(toOpenAITools(null), undefined);
    assert.equal(toOpenAITools(undefined), undefined);
    assert.equal(toOpenAITools([]), undefined);
  });

  test('converts a single tool to OpenAI function format', () => {
    const tools = [{
      name: 'get_weather',
      description: 'Get current weather',
      input_schema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    }];
    const result = toOpenAITools(tools);
    assert.equal(result.length, 1);
    assert.equal(result[0].type, 'function');
    assert.equal(result[0].function.name, 'get_weather');
    assert.equal(result[0].function.description, 'Get current weather');
    assert.deepEqual(result[0].function.parameters, tools[0].input_schema);
  });

  test('converts multiple tools preserving order', () => {
    const tools = [
      { name: 'tool_a', description: 'A', input_schema: {} },
      { name: 'tool_b', description: 'B', input_schema: {} },
    ];
    const result = toOpenAITools(tools);
    assert.equal(result.length, 2);
    assert.equal(result[0].function.name, 'tool_a');
    assert.equal(result[1].function.name, 'tool_b');
  });
});

// ── toOpenAIToolChoice ────────────────────────────────────────────────────────

describe('toOpenAIToolChoice', () => {
  test('returns undefined for null/undefined', () => {
    assert.equal(toOpenAIToolChoice(null), undefined);
    assert.equal(toOpenAIToolChoice(undefined), undefined);
  });

  test('maps auto → "auto"', () => {
    assert.equal(toOpenAIToolChoice({ type: 'auto' }), 'auto');
  });

  test('maps none → "none"', () => {
    assert.equal(toOpenAIToolChoice({ type: 'none' }), 'none');
  });

  test('maps any → "required"', () => {
    assert.equal(toOpenAIToolChoice({ type: 'any' }), 'required');
  });

  test('maps tool → {type:"function",function:{name}}', () => {
    const result = toOpenAIToolChoice({ type: 'tool', name: 'my_tool' });
    assert.deepEqual(result, { type: 'function', function: { name: 'my_tool' } });
  });

  test('returns undefined for unknown type', () => {
    assert.equal(toOpenAIToolChoice({ type: 'unknown' }), undefined);
  });
});

// ── toOpenAIMessages ──────────────────────────────────────────────────────────

describe('toOpenAIMessages', () => {
  test('empty messages with no system → empty array', () => {
    assert.deepEqual(toOpenAIMessages([], null), []);
    assert.deepEqual(toOpenAIMessages([], undefined), []);
  });

  test('adds system message for string system prompt', () => {
    const result = toOpenAIMessages([], 'You are helpful');
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'system');
    assert.equal(result[0].content, 'You are helpful');
  });

  test('adds system message for array system prompt with text block', () => {
    const system = [{ type: 'text', text: 'Be concise' }];
    const result = toOpenAIMessages([], system);
    assert.equal(result[0].role, 'system');
    assert.equal(result[0].content, 'Be concise');
  });

  test('adds system message for array with document block', () => {
    const system = [
      { type: 'text', text: 'Context:' },
      { type: 'document', source: { type: 'text', data: 'doc content' } },
    ];
    const result = toOpenAIMessages([], system);
    assert.equal(result[0].role, 'system');
    assert.ok(result[0].content.includes('Context:'));
    assert.ok(result[0].content.includes('doc content'));
  });

  test('converts simple string-content messages', () => {
    const messages = [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi there' },
    ];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 2);
    assert.equal(result[0].content, 'hello');
    assert.equal(result[1].content, 'hi there');
  });

  test('converts text content blocks array to a string', () => {
    const messages = [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result[0].content, 'hello');
  });

  test('joins multiple text blocks', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'text', text: 'first' },
        { type: 'text', text: 'second' },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result[0].content, 'firstsecond');
  });

  test('converts tool_result block to role:tool message', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'toolu_001',
        content: [{ type: 'text', text: '{"temp": 72}' }],
      }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 1);
    assert.equal(result[0].role, 'tool');
    assert.equal(result[0].tool_call_id, 'toolu_001');
    assert.equal(result[0].content, '{"temp": 72}');
  });

  test('prefixes tool_result error content with [ERROR]', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'toolu_001',
        is_error: true,
        content: [{ type: 'text', text: 'tool failed' }],
      }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.ok(result[0].content.startsWith('[ERROR]'));
    assert.ok(result[0].content.includes('tool failed'));
  });

  test('skips tool_result with missing tool_use_id', () => {
    const messages = [{
      role: 'user',
      content: [{ type: 'tool_result', content: [{ type: 'text', text: 'result' }] }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 0);
  });

  test('handles string-content tool_result', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'toolu_002',
        content: 'raw string result',
      }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result[0].content, 'raw string result');
  });

  test('tool_result with image content appends follow-up user message', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'toolu_003',
        content: [
          { type: 'text', text: 'screenshot captured' },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc123' } },
        ],
      }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 2);
    assert.equal(result[0].role, 'tool');
    assert.equal(result[0].content, 'screenshot captured');
    assert.equal(result[1].role, 'user');
    assert.ok(Array.isArray(result[1].content));
    assert.equal(result[1].content.length, 1);
    assert.equal(result[1].content[0].type, 'image_url');
    assert.equal(result[1].content[0].image_url.url, 'data:image/png;base64,abc123');
  });

  test('tool_result images and trailing text blocks merge into one user message', () => {
    const messages = [{
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: 'toolu_004',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'xyz' } },
          ],
        },
        { type: 'text', text: 'what do you see?' },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 2);
    assert.equal(result[0].role, 'tool');
    assert.equal(result[1].role, 'user');
    assert.ok(Array.isArray(result[1].content));
    assert.equal(result[1].content.length, 2);
    assert.equal(result[1].content[0].type, 'text');
    assert.equal(result[1].content[0].text, 'what do you see?');
    assert.equal(result[1].content[1].type, 'image_url');
  });

  test('converts tool_use blocks to tool_calls on assistant message', () => {
    const messages = [{
      role: 'assistant',
      content: [{
        type: 'tool_use',
        id: 'toolu_001',
        name: 'get_weather',
        input: { city: 'London' },
      }],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.ok(Array.isArray(result[0].tool_calls));
    assert.equal(result[0].tool_calls.length, 1);
    assert.equal(result[0].tool_calls[0].id, 'toolu_001');
    assert.equal(result[0].tool_calls[0].type, 'function');
    assert.equal(result[0].tool_calls[0].function.name, 'get_weather');
    assert.equal(result[0].tool_calls[0].function.arguments, JSON.stringify({ city: 'London' }));
  });

  test('converts thinking blocks to <think> tags', () => {
    const messages = [{
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'my reasoning' },
        { type: 'text', text: 'my answer' },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(typeof result[0].content, 'string');
    assert.ok(result[0].content.includes('<think>my reasoning</think>'));
    assert.ok(result[0].content.includes('my answer'));
  });

  test('converts image blocks to multipart content array', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'text', text: 'describe this' },
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc' } },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.ok(Array.isArray(result[0].content));
    const textPart = result[0].content.find(p => p.type === 'text');
    assert.equal(textPart?.text, 'describe this');
    const imgPart = result[0].content.find(p => p.type === 'image_url');
    assert.equal(imgPart?.image_url.url, 'data:image/png;base64,abc');
  });

  test('document block in user message is converted to text', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'text', text: 'Summarise:' },
        { type: 'document', title: 'Report', source: { type: 'text', data: 'Q1 revenue: $1M' } },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.ok(result[0].content.includes('Summarise:'));
    assert.ok(result[0].content.includes('[Document: Report]'));
    assert.ok(result[0].content.includes('Q1 revenue: $1M'));
  });

  test('preserves system + user + assistant order', () => {
    const result = toOpenAIMessages(
      [{ role: 'user', content: 'q' }, { role: 'assistant', content: 'a' }],
      'sys'
    );
    assert.equal(result[0].role, 'system');
    assert.equal(result[1].role, 'user');
    assert.equal(result[2].role, 'assistant');
  });
});

// ── logRequest ────────────────────────────────────────────────────────────────

describe('logRequest', () => {
  // Minimal req/res mocks that satisfy logRequest's access patterns.
  function mockReqRes({ method = 'POST', url = '/v1/messages', statusCode = 200, requestId = 'req_test123' } = {}) {
    const req = { method, url };
    const res = { statusCode, getHeader: (h) => h === 'request-id' ? requestId : undefined };
    return { req, res };
  }

  function captureLog(fn) {
    const lines = [];
    const orig = console.log;
    console.log = (...args) => lines.push(args.join(' '));
    try { fn(); } finally { console.log = orig; }
    return lines;
  }

  test('text format: includes method, url, status, and ms', () => {
    const { req, res } = mockReqRes();
    const lines = captureLog(() => logRequest(req, res, '/v1/messages', 123, null, 'text'));
    assert.equal(lines.length, 1);
    assert.ok(lines[0].includes('POST'), `missing method in: ${lines[0]}`);
    assert.ok(lines[0].includes('200'), `missing status in: ${lines[0]}`);
    assert.ok(lines[0].includes('123ms'), `missing duration in: ${lines[0]}`);
  });

  test('text format: includes token counts and model when meta is provided', () => {
    const { req, res } = mockReqRes();
    const meta = { model: 'qwen2.5:7b', tokensIn: 500, tokensOut: 200 };
    const lines = captureLog(() => logRequest(req, res, '/v1/messages', 99, meta, 'text'));
    assert.ok(lines[0].includes('in=500'), `missing tokens_in in: ${lines[0]}`);
    assert.ok(lines[0].includes('out=200'), `missing tokens_out in: ${lines[0]}`);
    assert.ok(lines[0].includes('model=qwen2.5:7b'), `missing model in: ${lines[0]}`);
  });

  test('text format: no token suffix when meta is null', () => {
    const { req, res } = mockReqRes({ method: 'GET', url: '/health', statusCode: 200 });
    const lines = captureLog(() => logRequest(req, res, '/health', 5, null, 'text'));
    assert.ok(!lines[0].includes('in='), `unexpected token info in: ${lines[0]}`);
  });

  test('json format: emits valid JSON with required fields', () => {
    const { req, res } = mockReqRes({ statusCode: 200, requestId: 'req_abc' });
    const lines = captureLog(() => logRequest(req, res, '/v1/messages', 77, null, 'json'));
    assert.equal(lines.length, 1);
    let parsed;
    assert.doesNotThrow(() => { parsed = JSON.parse(lines[0]); }, 'log line must be valid JSON');
    assert.equal(parsed.method, 'POST');
    assert.equal(parsed.status, 200);
    assert.equal(parsed.ms, 77);
    assert.equal(parsed.path, '/v1/messages');
    assert.equal(parsed.request_id, 'req_abc');
    assert.ok(typeof parsed.ts === 'string', 'ts must be an ISO string');
  });

  test('json format: includes model and token counts from meta', () => {
    const { req, res } = mockReqRes();
    const meta = { model: 'qwen2.5:14b', tokensIn: 1000, tokensOut: 400 };
    const lines = captureLog(() => logRequest(req, res, '/v1/messages', 300, meta, 'json'));
    const parsed = JSON.parse(lines[0]);
    assert.equal(parsed.model, 'qwen2.5:14b');
    assert.equal(parsed.tokens_in, 1000);
    assert.equal(parsed.tokens_out, 400);
  });

  test('json format: omits model/token fields when meta is null', () => {
    const { req, res } = mockReqRes({ method: 'GET', url: '/health' });
    const lines = captureLog(() => logRequest(req, res, '/health', 3, null, 'json'));
    const parsed = JSON.parse(lines[0]);
    assert.ok(!('model' in parsed), 'model should be absent');
    assert.ok(!('tokens_in' in parsed), 'tokens_in should be absent');
    assert.ok(!('tokens_out' in parsed), 'tokens_out should be absent');
  });
});

// ── getOllamaHost / OLLAMA_HOSTS ──────────────────────────────────────────────

describe('getOllamaHost', () => {
  test('OLLAMA_HOSTS is a non-empty array of strings', () => {
    assert.ok(Array.isArray(OLLAMA_HOSTS));
    assert.ok(OLLAMA_HOSTS.length >= 1);
    for (const h of OLLAMA_HOSTS) assert.equal(typeof h, 'string');
  });

  test('returns a URL-like string starting with http', () => {
    const host = getOllamaHost();
    assert.ok(host.startsWith('http'), `expected http URL, got: ${host}`);
  });

  test('always returns one of the configured hosts', () => {
    // Call several times and verify every result is in the OLLAMA_HOSTS list.
    for (let i = 0; i < OLLAMA_HOSTS.length * 3; i++) {
      const h = getOllamaHost();
      assert.ok(OLLAMA_HOSTS.includes(h), `unexpected host returned: ${h}`);
    }
  });
});

// ── checkRateLimit ────────────────────────────────────────────────────────────

describe('checkRateLimit', () => {
  // Build a minimal mock res that captures setHeader and writeHead/end calls.
  function mockRes() {
    const headers = {};
    return {
      headers,
      headersSent: false,
      writableEnded: false,
      statusCode: null,
      body: null,
      setHeader(k, v) { headers[k.toLowerCase()] = v; },
      writeHead(code) { this.statusCode = code; this.headersSent = true; },
      end(b) { this.body = b; this.writableEnded = true; },
    };
  }
  function mockReq() { return { headers: {}, socket: { remoteAddress: '127.0.0.1' } }; }

  // Ensure a fresh window key per test by using unique keys.
  let keyCounter = 0;
  function freshKey() { return `test-key-${++keyCounter}`; }

  test('returns true and sets x-ratelimit headers when under limit', () => {
    const req = mockReq();
    const res = mockRes();
    const ok = checkRateLimit(freshKey(), 10, req, res);
    assert.ok(ok, 'should return true when under limit');
    assert.equal(res.headers['x-ratelimit-limit-requests'], '10');
    assert.equal(res.headers['x-ratelimit-remaining-requests'], '9');
    assert.ok(res.headers['x-ratelimit-reset-requests'], 'reset header should be set');
    assert.equal(res.statusCode, null, 'should not write a status code when allowed');
  });

  test('returns false and writes 429 when over limit', () => {
    const req = mockReq();
    const key = freshKey();
    // Exhaust the limit of 2 with two allowed requests.
    checkRateLimit(key, 2, req, mockRes());
    checkRateLimit(key, 2, req, mockRes());
    // Third request should be rejected.
    const res = mockRes();
    const ok = checkRateLimit(key, 2, req, res);
    assert.ok(!ok, 'should return false when over limit');
    assert.equal(res.statusCode, 429);
    assert.ok(res.body, 'should write a response body');
    const body = JSON.parse(res.body);
    assert.equal(body.error.type, 'rate_limit_error');
    assert.ok(body.error.message.includes('req/min'), 'message should mention rate');
    assert.ok(res.headers['retry-after'], 'should set retry-after header');
  });

  test('remaining decrements on each call', () => {
    const req = mockReq();
    const key = freshKey();
    const limit = 5;
    for (let i = 0; i < limit; i++) {
      const res = mockRes();
      checkRateLimit(key, limit, req, res);
      assert.equal(res.headers['x-ratelimit-remaining-requests'], String(limit - i - 1));
    }
  });

  test('window resets after 60 seconds (simulated via _rateLimitWindows)', () => {
    const req = mockReq();
    const key = freshKey();
    // Exhaust the limit.
    checkRateLimit(key, 1, req, mockRes());
    const res1 = mockRes();
    assert.ok(!checkRateLimit(key, 1, req, res1), 'should be rate-limited');
    // Wind back the window start to simulate expiry.
    const w = _rateLimitWindows.get(key);
    w.windowStart -= 61_000;
    // Next request should start a fresh window and succeed.
    const res2 = mockRes();
    assert.ok(checkRateLimit(key, 1, req, res2), 'should succeed after window reset');
    assert.equal(res2.headers['x-ratelimit-remaining-requests'], '0');
  });
});

// ── getClientIp ───────────────────────────────────────────────────────────────

describe('getClientIp', () => {
  test('returns socket remoteAddress when no x-forwarded-for', () => {
    const req = { headers: {}, socket: { remoteAddress: '10.0.0.1' } };
    assert.equal(getClientIp(req), '10.0.0.1');
  });

  test('returns first IP from x-forwarded-for header', () => {
    const req = { headers: { 'x-forwarded-for': '203.0.113.5, 10.0.0.2, 10.0.0.3' }, socket: { remoteAddress: '10.0.0.2' } };
    assert.equal(getClientIp(req), '203.0.113.5');
  });

  test('trims whitespace from x-forwarded-for', () => {
    const req = { headers: { 'x-forwarded-for': '  192.168.1.1  ' }, socket: { remoteAddress: '10.0.0.1' } };
    assert.equal(getClientIp(req), '192.168.1.1');
  });

  test('falls back to "unknown" when socket has no remoteAddress', () => {
    const req = { headers: {}, socket: {} };
    assert.equal(getClientIp(req), 'unknown');
  });
});
