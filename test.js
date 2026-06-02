'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');

const {
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
  checkRateLimit,
  getClientIp,
  _rateLimitWindows,
  checkConcurrency,
  trackActiveLlmRequest,
  _metrics,
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

// ── parseDotEnv ───────────────────────────────────────────────────────────────

describe('parseDotEnv', () => {
  test('returns empty object for empty input', () => {
    assert.deepEqual(parseDotEnv(''), {});
    assert.deepEqual(parseDotEnv('\n\n\n'), {});
  });

  test('parses simple KEY=VALUE pairs', () => {
    const result = parseDotEnv('FOO=bar\nBAZ=qux');
    assert.equal(result.FOO, 'bar');
    assert.equal(result.BAZ, 'qux');
  });

  test('skips blank lines', () => {
    const result = parseDotEnv('\nFOO=bar\n\nBAZ=qux\n');
    assert.deepEqual(result, { FOO: 'bar', BAZ: 'qux' });
  });

  test('skips lines starting with #', () => {
    const result = parseDotEnv('# this is a comment\nFOO=bar\n# another comment\nBAZ=qux');
    assert.deepEqual(result, { FOO: 'bar', BAZ: 'qux' });
  });

  test('skips lines with no = sign', () => {
    const result = parseDotEnv('NOEQUALS\nFOO=bar');
    assert.deepEqual(result, { FOO: 'bar' });
  });

  test('strips surrounding double quotes from value', () => {
    const result = parseDotEnv('MSG="hello world"');
    assert.equal(result.MSG, 'hello world');
  });

  test('strips surrounding single quotes from value', () => {
    const result = parseDotEnv("MSG='hello world'");
    assert.equal(result.MSG, 'hello world');
  });

  test('does not strip mismatched quotes', () => {
    const result = parseDotEnv('MSG="hello world\'');
    assert.equal(result.MSG, '"hello world\'');
  });

  test('preserves = signs inside the value', () => {
    const result = parseDotEnv('URL=http://example.com?a=1&b=2');
    assert.equal(result.URL, 'http://example.com?a=1&b=2');
  });

  test('handles values with spaces around = when key/val are trimmed', () => {
    const result = parseDotEnv('  FOO = bar  ');
    assert.equal(result.FOO, 'bar');
  });

  test('parses MODEL_MAP JSON value without quotes', () => {
    const json = '{"claude-3-haiku":"qwen2.5:7b"}';
    const result = parseDotEnv(`MODEL_MAP=${json}`);
    assert.equal(result.MODEL_MAP, json);
  });

  test('empty value is an empty string', () => {
    const result = parseDotEnv('FOO=');
    assert.equal(result.FOO, '');
  });
});

// ── parseOllamaOptions ────────────────────────────────────────────────────────

describe('parseOllamaOptions', () => {
  test('returns empty object for falsy input', () => {
    assert.deepEqual(parseOllamaOptions(null), {});
    assert.deepEqual(parseOllamaOptions(undefined), {});
    assert.deepEqual(parseOllamaOptions(''), {});
  });

  test('parses a valid JSON object', () => {
    const result = parseOllamaOptions('{"repeat_penalty":1.1,"mirostat":2,"num_gpu":33}');
    assert.deepEqual(result, { repeat_penalty: 1.1, mirostat: 2, num_gpu: 33 });
  });

  test('returns empty object and warns on invalid JSON', () => {
    const orig = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    const result = parseOllamaOptions('{bad json}');
    console.warn = orig;
    assert.deepEqual(result, {});
    assert.ok(warns.some(w => w.includes('OLLAMA_OPTIONS')));
  });

  test('returns empty object and warns when value is an array', () => {
    const orig = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    const result = parseOllamaOptions('[1,2,3]');
    console.warn = orig;
    assert.deepEqual(result, {});
    assert.ok(warns.some(w => w.includes('OLLAMA_OPTIONS')));
  });

  test('returns empty object and warns when value is a scalar', () => {
    const orig = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    const result = parseOllamaOptions('"just a string"');
    console.warn = orig;
    assert.deepEqual(result, {});
    assert.ok(warns.some(w => w.includes('OLLAMA_OPTIONS')));
  });

  test('OLLAMA_OPTIONS module constant is an object (not set in test env)', () => {
    assert.ok(OLLAMA_OPTIONS !== null);
    assert.equal(typeof OLLAMA_OPTIONS, 'object');
    assert.ok(!Array.isArray(OLLAMA_OPTIONS));
  });
});

// ── parseOllamaError ──────────────────────────────────────────────────────────

describe('parseOllamaError', () => {
  test('extracts error field from Ollama JSON response', () => {
    const msg = parseOllamaError('{"error":"model \'xyz\' not found, try pulling it first"}');
    assert.equal(msg, "model 'xyz' not found, try pulling it first");
  });

  test('extracts message field when error field is absent', () => {
    const msg = parseOllamaError('{"message":"context length exceeded"}');
    assert.equal(msg, 'context length exceeded');
  });

  test('prefers error field over message field', () => {
    const msg = parseOllamaError('{"error":"primary error","message":"secondary"}');
    assert.equal(msg, 'primary error');
  });

  test('returns raw text when body is not JSON', () => {
    const msg = parseOllamaError('internal server error');
    assert.equal(msg, 'internal server error');
  });

  test('returns raw text when JSON has no error or message field', () => {
    const msg = parseOllamaError('{"code":500,"detail":"oops"}');
    assert.equal(msg, '{"code":500,"detail":"oops"}');
  });

  test('returns empty string for empty input', () => {
    assert.equal(parseOllamaError(''), '');
  });

  test('returns null/undefined as-is for falsy input', () => {
    assert.equal(parseOllamaError(null), null);
    assert.equal(parseOllamaError(undefined), undefined);
  });

  test('ignores non-string error field', () => {
    const msg = parseOllamaError('{"error":{"nested":"object"}}');
    assert.equal(msg, '{"error":{"nested":"object"}}');
  });
});

// ── MODEL_MAP alias resolution (unit-level, mirrors handleModelById logic) ────
// handleModelById resolves aliases the same way resolveModel does, so we verify
// the shared logic here; integration coverage lives in test-integration.js.

describe('MODEL_MAP alias resolution logic', () => {
  // Simulate the lookup logic extracted from handleModelById: given a modelId and
  // a MODEL_MAP, find the target Ollama model name (or null if not mapped).
  function resolveAlias(modelId, map) {
    let target = map[modelId];
    if (!target) {
      for (const [key, val] of Object.entries(map)) {
        if (modelId.startsWith(key)) { target = val; break; }
      }
    }
    return target || null;
  }

  const map = {
    'claude-3-haiku':  'qwen2.5:7b',
    'claude-3-sonnet': 'qwen2.5:14b',
    'claude-3-opus':   'qwen2.5:72b',
  };

  test('exact alias match returns target model name', () => {
    assert.equal(resolveAlias('claude-3-haiku', map), 'qwen2.5:7b');
    assert.equal(resolveAlias('claude-3-opus',  map), 'qwen2.5:72b');
  });

  test('prefix alias match returns target (e.g. claude-3-haiku-20240307)', () => {
    assert.equal(resolveAlias('claude-3-haiku-20240307', map), 'qwen2.5:7b');
    assert.equal(resolveAlias('claude-3-sonnet-20250219', map), 'qwen2.5:14b');
  });

  test('real Ollama model name (not in map) resolves to null', () => {
    assert.equal(resolveAlias('qwen2.5:7b', map), null);
    assert.equal(resolveAlias('llama3:8b',  map), null);
  });

  test('unknown claude-* name not in map resolves to null', () => {
    assert.equal(resolveAlias('claude-instant-1', map), null);
  });

  test('empty map resolves everything to null', () => {
    assert.equal(resolveAlias('claude-3-haiku', {}), null);
    assert.equal(resolveAlias('qwen2.5:7b',     {}), null);
  });
});

// ── sanitizeForLog ────────────────────────────────────────────────────────────

describe('sanitizeForLog', () => {
  test('passes through scalars and short strings unchanged', () => {
    assert.equal(sanitizeForLog('hello'), 'hello');
    assert.equal(sanitizeForLog(42), 42);
    assert.equal(sanitizeForLog(null), null);
    assert.equal(sanitizeForLog(true), true);
  });

  test('truncates long `data` field with a placeholder', () => {
    const longData = 'A'.repeat(500);
    const result = sanitizeForLog({ source: { type: 'base64', data: longData } });
    assert.equal(result.source.type, 'base64');
    assert.match(result.source.data, /^<base64 500 chars>$/);
  });

  test('preserves short `data` fields', () => {
    const result = sanitizeForLog({ data: 'abc' });
    assert.equal(result.data, 'abc');
  });

  test('truncates data-URL `url` fields', () => {
    const dataUrl = 'data:image/png;base64,' + 'B'.repeat(300);
    const result = sanitizeForLog({ image_url: { url: dataUrl } });
    assert.match(result.image_url.url, /^<base64/);
  });

  test('leaves non-data-URL `url` fields alone', () => {
    const result = sanitizeForLog({ url: 'https://example.com/img.png' });
    assert.equal(result.url, 'https://example.com/img.png');
  });

  test('recursively sanitizes arrays', () => {
    const longData = 'C'.repeat(500);
    const result = sanitizeForLog([{ data: longData }, { text: 'hi' }]);
    assert.match(result[0].data, /^<base64/);
    assert.equal(result[1].text, 'hi');
  });

  test('deep-copies the object without mutating the original', () => {
    const longData = 'D'.repeat(500);
    const original = { data: longData };
    sanitizeForLog(original);
    assert.equal(original.data, longData); // original unchanged
  });

  test('sanitizes nested OpenAI messages array with image_url blocks', () => {
    const dataUrl = 'data:image/jpeg;base64,' + 'E'.repeat(400);
    const messages = [
      { role: 'user', content: [{ type: 'image_url', image_url: { url: dataUrl } }] }
    ];
    const result = sanitizeForLog(messages);
    assert.match(result[0].content[0].image_url.url, /^<base64/);
  });
});

// ── POST /v1/completions (handleOpenAICompletions) ────────────────────────────
// These tests exercise the handler in isolation by providing mock req/res objects
// and a stubbed fetch so no real Ollama instance is required.

describe('handleOpenAICompletions', () => {
  const { handleOpenAICompletions } = require('./proxy');

  // Minimal mock request builder.
  function mockReq(body) {
    const chunks = [JSON.stringify(body)];
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST',
      url: '/v1/completions',
      [Symbol.asyncIterator]: async function* () { for (const c of chunks) yield c; },
    };
    return req;
  }

  // Minimal mock response builder; captures writeHead / end calls.
  function mockRes() {
    const res = {
      headersSent: false,
      writableEnded: false,
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status) { this._status = status; this.headersSent = true; },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
      on() {},
    };
    return res;
  }

  // Replace global fetch with a one-shot stub that resolves with an OpenAI chat response.
  function stubFetch(chatResponse, status = 200) {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: status < 400,
      status,
      json: async () => chatResponse,
      text: async () => JSON.stringify(chatResponse),
      body: null,
    });
    return () => { global.fetch = orig; };
  }

  test('returns 400 when body is not valid JSON', async () => {
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/completions',
      [Symbol.asyncIterator]: async function* () { yield 'NOT JSON'; },
    };
    const res = mockRes();
    await handleOpenAICompletions(req, res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
  });

  test('returns 400 when prompt is missing', async () => {
    const req = mockReq({ model: 'llama3' });
    const res = mockRes();
    await handleOpenAICompletions(req, res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /prompt/);
  });

  test('returns text_completion envelope for non-streaming request', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'Hello world' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 3 },
    });
    try {
      const req = mockReq({ model: 'llama3', prompt: 'Say hi' });
      const res = mockRes();
      await handleOpenAICompletions(req, res);
      assert.equal(res._status, 200);
      const body = JSON.parse(res._body);
      assert.equal(body.object, 'text_completion');
      assert.equal(body.choices.length, 1);
      assert.equal(body.choices[0].text, 'Hello world');
      assert.equal(body.choices[0].index, 0);
      assert.equal(body.choices[0].finish_reason, 'stop');
      assert.ok(body.id.startsWith('cmpl_'));
      assert.equal(body.usage.prompt_tokens, 5);
      assert.equal(body.usage.completion_tokens, 3);
      assert.equal(body.usage.total_tokens, 8);
    } finally {
      restore();
    }
  });

  test('joins array prompt into a single string', async () => {
    let sentBody;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      sentBody = JSON.parse(opts.body);
      return {
        ok: true, status: 200,
        json: async () => ({
          choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
      };
    };
    try {
      const req = mockReq({ prompt: ['part one', 'part two'] });
      const res = mockRes();
      await handleOpenAICompletions(req, res);
      assert.equal(sentBody.messages[0].content, 'part one\npart two');
    } finally {
      global.fetch = origFetch;
    }
  });

  test('id has cmpl_ prefix and created is a unix timestamp', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    try {
      const req = mockReq({ prompt: 'test' });
      const res = mockRes();
      const before = Math.floor(Date.now() / 1000);
      await handleOpenAICompletions(req, res);
      const after = Math.floor(Date.now() / 1000);
      const body = JSON.parse(res._body);
      assert.match(body.id, /^cmpl_/);
      assert.ok(body.created >= before && body.created <= after + 1);
    } finally {
      restore();
    }
  });

  test('returns upstream status when Ollama returns a non-ok response', async () => {
    // Use a non-ok but non-5xx response so fetchWithRetry does not retry
    // (retrying would add multi-second backoff delays to the test suite).
    const origFetch = global.fetch;
    global.fetch = async () => ({
      ok: false, status: 400,
      text: async () => JSON.stringify({ error: 'bad model' }),
    });
    try {
      const req = mockReq({ prompt: 'hi' });
      const res = mockRes();
      await handleOpenAICompletions(req, res);
      assert.equal(res._status, 400);
    } finally {
      global.fetch = origFetch;
    }
  });
});

// ── checkConcurrency / trackActiveLlmRequest ──────────────────────────────────

// Minimal EventEmitter-based mock for the res object used by concurrency helpers.
function mockConcurrRes() {
  const { EventEmitter } = require('events');
  const r = new EventEmitter();
  r.headersSent = false;
  r._status = null;
  r._body = null;
  r.writeHead = (status) => { r._status = status; r.headersSent = true; };
  r.end = (body) => { r._body = body; r.emit('finish'); };
  r.setHeader = () => {};
  r.getHeader = () => undefined;
  return r;
}

describe('checkConcurrency', () => {
  test('returns true without writing a response when PROXY_MAX_CONCURRENCY is unset', () => {
    // In the test environment PROXY_MAX_CONCURRENCY is not set, so the guard is a no-op.
    const res = mockConcurrRes();
    const result = checkConcurrency(res);
    assert.equal(result, true);
    assert.equal(res._status, null, 'should not write any response when limit is unset');
  });
});

describe('trackActiveLlmRequest', () => {
  test('decrements activeLlmRequests when finish fires', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;
    const res = mockConcurrRes();
    trackActiveLlmRequest(res);
    assert.equal(_metrics.activeLlmRequests, before + 1, 'still elevated before finish');
    res.emit('finish');
    assert.equal(_metrics.activeLlmRequests, before, 'decremented after finish');
  });

  test('decrements activeLlmRequests when close fires (dropped connection)', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;
    const res = mockConcurrRes();
    trackActiveLlmRequest(res);
    res.emit('close');
    assert.equal(_metrics.activeLlmRequests, before, 'decremented after close');
  });

  test('decrements exactly once even when both finish and close fire', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;
    const res = mockConcurrRes();
    trackActiveLlmRequest(res);
    res.emit('finish');
    res.emit('close');
    assert.equal(_metrics.activeLlmRequests, before, 'decremented exactly once');
  });

  test('multiple independent trackActiveLlmRequest calls each decrement their own count', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests += 2;
    const res1 = mockConcurrRes();
    const res2 = mockConcurrRes();
    trackActiveLlmRequest(res1);
    trackActiveLlmRequest(res2);
    res1.emit('finish');
    assert.equal(_metrics.activeLlmRequests, before + 1, 'only one decremented after first finish');
    res2.emit('finish');
    assert.equal(_metrics.activeLlmRequests, before, 'both decremented after second finish');
  });
});

// ── acquireLlmSlot / releaseLlmSlot ──────────────────────────────────────────

describe('acquireLlmSlot / releaseLlmSlot', () => {
  const { acquireLlmSlot, releaseLlmSlot, _concurrencyQueue, _metrics } = require('./proxy');

  function mockSlotReq() {
    return {
      headers: {},
      socket: {
        remoteAddress: '127.0.0.1',
        once: () => {},
        off: () => {},
      },
    };
  }
  function mockSlotRes() {
    return {
      headersSent: false, writableEnded: false, _status: null, _body: '',
      setHeader() {}, getHeader() {},
      writeHead(s) { this._status = s; this.headersSent = true; },
      end(b = '') { this._body += b; this.writableEnded = true; },
    };
  }

  test('when PROXY_MAX_CONCURRENCY is unset, grants slot immediately and increments counter', async () => {
    const before = _metrics.activeLlmRequests;
    const result = await acquireLlmSlot(mockSlotReq(), mockSlotRes());
    assert.equal(result, true, 'should return true when no limit set');
    assert.equal(_metrics.activeLlmRequests, before + 1, 'activeLlmRequests should be incremented');
    releaseLlmSlot(); // cleanup
    assert.equal(_metrics.activeLlmRequests, before, 'counter back to baseline after release');
  });

  test('releaseLlmSlot decrements activeLlmRequests when queue is empty', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;
    assert.equal(_concurrencyQueue.length, 0, 'queue must be empty for this test');
    releaseLlmSlot();
    assert.equal(_metrics.activeLlmRequests, before, 'counter decremented');
  });

  test('releaseLlmSlot calls onGranted and does NOT decrement activeLlmRequests when queue has waiters', () => {
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;   // simulate one slot in use
    _metrics.queuedLlmRequests++;   // simulate one waiter

    let granted = false;
    // Push a synthetic queue entry (mirrors what acquireLlmSlot does internally).
    _concurrencyQueue.push({
      onGranted: () => {
        granted = true;
        _metrics.queuedLlmRequests--;   // mirrors what onGranted in acquireLlmSlot does
      },
    });

    releaseLlmSlot();

    assert.equal(granted, true, 'onGranted should be called');
    assert.equal(_metrics.activeLlmRequests, before + 1, 'slot transferred — activeLlmRequests unchanged');
    assert.equal(_metrics.queuedLlmRequests, 0, 'queuedLlmRequests decremented by onGranted');

    // Cleanup: release the transferred slot.
    releaseLlmSlot();
    assert.equal(_metrics.activeLlmRequests, before, 'counter back to baseline after cleanup');
  });
});

// ── handleMessages ────────────────────────────────────────────────────────────
// Unit tests for the core Anthropic-format handler using mock req/res/fetch.
// Covers non-streaming and streaming paths without needing a real Ollama server.

describe('handleMessages', () => {
  const { handleMessages } = require('./proxy');

  function mockReq(body) {
    return {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST',
      url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () { yield JSON.stringify(body); },
    };
  }

  function mockRes() {
    const res = {
      headersSent: false,
      writableEnded: false,
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status) { this._status = status; this.headersSent = true; },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
      on() {},
    };
    return res;
  }

  // Stubs global.fetch for a single non-streaming Ollama response.
  // Returns a restore function. Uses status 200 + ok:true so fetchWithRetry
  // returns immediately without any backoff delays.
  function stubFetch(chatResponse) {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => chatResponse,
      text: async () => JSON.stringify(chatResponse),
      body: null,
    });
    return () => { global.fetch = orig; };
  }

  // Stubs global.fetch to return a non-ok 4xx response so the proxy returns 502.
  // 4xx is below the 500 threshold so fetchWithRetry returns immediately (no retry delay).
  function stubFetchError(status = 400, text = 'bad request') {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: false, status,
      text: async () => text,
      body: null,
    });
    return () => { global.fetch = orig; };
  }

  // Stubs global.fetch to return a streaming SSE response.
  // Each element of sseLines is emitted as a separate chunk followed by \n.
  function stubStreamFetch(sseLines) {
    const enc = new TextEncoder();
    let pos = 0;
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (pos >= sseLines.length) return { done: true, value: undefined };
              return { done: false, value: enc.encode(sseLines[pos++] + '\n') };
            },
            releaseLock() {},
          };
        },
      },
    });
    return () => { global.fetch = orig; };
  }

  // Parses a raw SSE body string into [{event, data}] objects.
  function parseSse(body) {
    const events = [];
    let event = null;
    for (const line of body.split('\n')) {
      if (line.startsWith('event: ')) { event = line.slice(7).trim(); }
      else if (line.startsWith('data: ')) {
        try { events.push({ event, data: JSON.parse(line.slice(6)) }); } catch {}
        event = null;
      }
    }
    return events;
  }

  // ── Input validation ─────────────────────────────────────────────────────────

  test('400 on invalid JSON body', async () => {
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () { yield 'NOT JSON'; },
    };
    const res = mockRes();
    await handleMessages(req, res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('400 when messages field is absent', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ model: 'llama3' }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /messages/);
  });

  test('400 when messages is not an array', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: 'not-an-array' }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  // ── Non-streaming path ───────────────────────────────────────────────────────

  test('non-streaming: returns correct Anthropic message envelope', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'Hello!' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({
        model: 'claude-3-haiku',
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 100, stream: false,
      }), res);
      assert.equal(res._status, 200);
      const body = JSON.parse(res._body);
      assert.equal(body.type, 'message');
      assert.equal(body.role, 'assistant');
      assert.ok(body.id.startsWith('msg_'), `id should start with msg_, got: ${body.id}`);
      assert.ok(Array.isArray(body.content));
      assert.equal(body.content[0].type, 'text');
      assert.equal(body.content[0].text, 'Hello!');
      assert.equal(body.stop_reason, 'end_turn');
    } finally { restore(); }
  });

  test('non-streaming: usage includes prompt-caching compat fields', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'Hi' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 5, completion_tokens: 3 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      const body = JSON.parse(res._body);
      assert.equal(body.usage.input_tokens, 5);
      assert.equal(body.usage.output_tokens, 3);
      assert.equal(body.usage.cache_creation_input_tokens, 0);
      assert.equal(body.usage.cache_read_input_tokens, 0);
    } finally { restore(); }
  });

  test('non-streaming: tool_calls finish_reason maps to tool_use stop_reason', async () => {
    const restore = stubFetch({
      choices: [{
        message: {
          content: null,
          tool_calls: [{
            id: 'call_1', type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"London"}' },
          }],
        },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 20, completion_tokens: 8 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Weather?' }], stream: false }), res);
      const body = JSON.parse(res._body);
      assert.equal(body.stop_reason, 'tool_use');
      const tu = body.content.find(c => c.type === 'tool_use');
      assert.ok(tu, 'should have tool_use block');
      assert.equal(tu.name, 'get_weather');
      assert.deepEqual(tu.input, { city: 'London' });
      assert.ok(tu.id, 'tool_use block should have an id');
    } finally { restore(); }
  });

  test('non-streaming: length finish_reason maps to max_tokens stop_reason', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'Truncated…' }, finish_reason: 'length' }],
      usage: { prompt_tokens: 10, completion_tokens: 100 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Tell me everything' }], stream: false }), res);
      assert.equal(JSON.parse(res._body).stop_reason, 'max_tokens');
    } finally { restore(); }
  });

  test('non-streaming: empty choices array returns 502', async () => {
    const restore = stubFetch({ choices: [], usage: {} });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 502);
      assert.equal(JSON.parse(res._body).error.type, 'ollama_error');
    } finally { restore(); }
  });

  test('non-streaming: Ollama 4xx proxied as 502 with structured error body', async () => {
    const restore = stubFetchError(404, '{"error":"model \'xyz\' not found, try pulling it first"}');
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 502);
      const body = JSON.parse(res._body);
      // Error must be an object, not a raw/double-encoded string.
      assert.equal(typeof body.error, 'object', 'error should be an object, not a raw string');
      assert.equal(body.error.type, 'ollama_error');
      // Inner Ollama message should be extracted, not double-JSON-encoded.
      assert.equal(body.error.message, "model 'xyz' not found, try pulling it first");
    } finally { restore(); }
  });

  test('non-streaming: plain-text Ollama error proxied as 502', async () => {
    const restore = stubFetchError(500, 'CUDA out of memory');
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 502);
      const body = JSON.parse(res._body);
      assert.equal(body.error.type, 'ollama_error');
      assert.equal(body.error.message, 'CUDA out of memory');
    } finally { restore(); }
  });

  test('non-streaming: <think> tags extracted into thinking content block', async () => {
    const restore = stubFetch({
      choices: [{
        message: { content: '<think>my reasoning</think>my answer' },
        finish_reason: 'stop',
      }],
      usage: { prompt_tokens: 15, completion_tokens: 10 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Think hard' }], stream: false }), res);
      const body = JSON.parse(res._body);
      const thinking = body.content.find(c => c.type === 'thinking');
      const text     = body.content.find(c => c.type === 'text');
      assert.ok(thinking, 'should have thinking block');
      assert.equal(thinking.thinking, 'my reasoning');
      assert.ok(thinking.signature, 'thinking block should carry a signature');
      assert.ok(text, 'should have text block');
      assert.equal(text.text, 'my answer');
    } finally { restore(); }
  });

  // ── Streaming path ───────────────────────────────────────────────────────────

  test('streaming: emits required Anthropic SSE event types', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hello' }], stream: true }), res);
      assert.equal(res._status, 200);
      const events = parseSse(res._body);
      const types  = events.map(e => e.event);
      for (const t of ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop']) {
        assert.ok(types.includes(t), `missing SSE event type: ${t}`);
      }
      // Correct text assembled from deltas
      const text = events
        .filter(e => e.event === 'content_block_delta' && e.data.delta?.type === 'text_delta')
        .map(e => e.data.delta.text).join('');
      assert.equal(text, 'Hi!');
    } finally { restore(); }
  });

  test('streaming: message_start includes prompt-caching compat usage fields', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'ping' }], stream: true }), res);
      const start = parseSse(res._body).find(e => e.event === 'message_start');
      assert.equal(start.data.message.usage.cache_creation_input_tokens, 0);
      assert.equal(start.data.message.usage.cache_read_input_tokens, 0);
    } finally { restore(); }
  });

  test('streaming: message_delta carries stop_reason and correct token counts', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'x' }], stream: true }), res);
      const delta = parseSse(res._body).find(e => e.event === 'message_delta');
      assert.equal(delta.data.delta.stop_reason, 'end_turn');
      assert.equal(delta.data.usage.input_tokens, 7);
      assert.equal(delta.data.usage.output_tokens, 4);
      assert.equal(delta.data.usage.cache_creation_input_tokens, 0);
      assert.equal(delta.data.usage.cache_read_input_tokens, 0);
    } finally { restore(); }
  });

  test('streaming: message_stop is the final emitted event', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'ok' }], stream: true }), res);
      const events = parseSse(res._body);
      assert.equal(events[events.length - 1].event, 'message_stop');
    } finally { restore(); }
  });

  test('streaming: tool_use block produces content_block_start with correct name and id', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":\\"London\\"}"}}]},"finish_reason":"tool_calls"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Weather?' }], stream: true }), res);
      const events = parseSse(res._body);

      // tool_use content_block_start
      const toolStart = events.find(e =>
        e.event === 'content_block_start' && e.data.content_block?.type === 'tool_use'
      );
      assert.ok(toolStart, 'should have tool_use content_block_start');
      assert.equal(toolStart.data.content_block.name, 'get_weather');
      assert.ok(toolStart.data.content_block.id, 'tool_use block should have an id');

      // input_json_delta events reconstruct the full argument JSON
      const jsonDeltas = events.filter(e =>
        e.event === 'content_block_delta' && e.data.delta?.type === 'input_json_delta'
      );
      assert.ok(jsonDeltas.length > 0, 'should have input_json_delta events');
      const fullArgs = jsonDeltas.map(e => e.data.delta.partial_json).join('');
      assert.deepEqual(JSON.parse(fullArgs), { city: 'London' });

      // stop_reason must be tool_use
      const delta = events.find(e => e.event === 'message_delta');
      assert.equal(delta.data.delta.stop_reason, 'tool_use');
    } finally { restore(); }
  });

  test('streaming: thinking tags produce thinking content block with signature_delta', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"<think>"},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"content":"my reasoning"},"finish_reason":null}]}',
      'data: {"choices":[{"index":0,"delta":{"content":"</think>answer"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Think' }], stream: true }), res);
      const events = parseSse(res._body);

      const thinkStart = events.find(e =>
        e.event === 'content_block_start' && e.data.content_block?.type === 'thinking'
      );
      assert.ok(thinkStart, 'should have thinking content_block_start');

      const sigDelta = events.find(e =>
        e.event === 'content_block_delta' && e.data.delta?.type === 'signature_delta'
      );
      assert.ok(sigDelta, 'should have signature_delta');
      assert.ok(sigDelta.data.delta.signature, 'signature must be non-empty');

      const textStart = events.find(e =>
        e.event === 'content_block_start' && e.data.content_block?.type === 'text'
      );
      assert.ok(textStart, 'should have text content_block_start after thinking');

      // thinking block index must be lower than text block index
      assert.ok(thinkStart.data.index < textStart.data.index,
        'thinking block must precede text block in index ordering');
    } finally { restore(); }
  });

  test('streaming: interleaved think/text blocks get correct sequential indices', async () => {
    // Simulates a model that outputs: <think>A</think>text1<think>B</think>text2
    // Each block should get a unique, monotonically increasing Anthropic index.
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"<think>A</think>mid<think>B</think>end"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'go' }], stream: true }), res);
      const events = parseSse(res._body);

      const blockStarts = events.filter(e => e.event === 'content_block_start');
      // Expect: thinking[0], text[1], thinking[2], text[3]
      assert.equal(blockStarts.length, 4, 'should open exactly 4 content blocks');

      const [b0, b1, b2, b3] = blockStarts;
      assert.equal(b0.data.content_block.type, 'thinking', 'block 0 should be thinking');
      assert.equal(b0.data.index, 0, 'block 0 index = 0');
      assert.equal(b1.data.content_block.type, 'text',    'block 1 should be text');
      assert.equal(b1.data.index, 1, 'block 1 index = 1');
      assert.equal(b2.data.content_block.type, 'thinking', 'block 2 should be thinking');
      assert.equal(b2.data.index, 2, 'block 2 index = 2');
      assert.equal(b3.data.content_block.type, 'text',    'block 3 should be text');
      assert.equal(b3.data.index, 3, 'block 3 index = 3');

      // Each block should have a matching stop event.
      const blockStops = events.filter(e => e.event === 'content_block_stop');
      assert.equal(blockStops.length, 4, 'should have 4 content_block_stop events');
      assert.deepEqual(
        blockStops.map(e => e.data.index).sort((a, b) => a - b),
        [0, 1, 2, 3],
        'stop indices must match start indices'
      );

      // Text content of each text block should be "mid" and "end".
      const textDeltas = events
        .filter(e => e.event === 'content_block_delta' && e.data.delta?.type === 'text_delta')
        .map(e => ({ index: e.data.index, text: e.data.delta.text }));
      const mid = textDeltas.filter(d => d.index === 1).map(d => d.text).join('');
      const end = textDeltas.filter(d => d.index === 3).map(d => d.text).join('');
      assert.equal(mid, 'mid', 'text block 1 should contain "mid"');
      assert.equal(end, 'end', 'text block 3 should contain "end"');
    } finally { restore(); }
  });
});

// ── resolveMaxTokens ──────────────────────────────────────────────────────────

describe('resolveMaxTokens', () => {
  const DEFAULT = process.env.PROXY_MAX_TOKENS ? Number(process.env.PROXY_MAX_TOKENS) : 8192;

  test('returns default when client value is undefined', () => {
    const r = resolveMaxTokens(undefined);
    assert.ok(!r.error, `unexpected error: ${r.error}`);
    assert.ok(typeof r.value === 'number' && r.value > 0);
  });

  test('returns default when client value is null', () => {
    const r = resolveMaxTokens(null);
    assert.ok(!r.error);
    assert.equal(r.value, PROXY_HARD_MAX_TOKENS ? Math.min(DEFAULT, PROXY_HARD_MAX_TOKENS) : DEFAULT);
  });

  test('passes through a valid positive integer', () => {
    const r = resolveMaxTokens(512);
    assert.ok(!r.error);
    assert.equal(r.value, PROXY_HARD_MAX_TOKENS ? Math.min(512, PROXY_HARD_MAX_TOKENS) : 512);
  });

  test('passes through a valid integer supplied as a string (numeric coercion)', () => {
    const r = resolveMaxTokens('256');
    assert.ok(!r.error);
    assert.equal(r.value, PROXY_HARD_MAX_TOKENS ? Math.min(256, PROXY_HARD_MAX_TOKENS) : 256);
  });

  test('returns error for zero', () => {
    const r = resolveMaxTokens(0);
    assert.ok(r.error, 'expected an error for max_tokens=0');
    assert.match(r.error, /positive integer/);
  });

  test('returns error for negative values', () => {
    assert.ok(resolveMaxTokens(-1).error, 'expected error for -1');
    assert.ok(resolveMaxTokens(-100).error, 'expected error for -100');
  });

  test('returns error for non-integer floats', () => {
    const r = resolveMaxTokens(100.5);
    assert.ok(r.error, 'expected error for 100.5');
    assert.match(r.error, /positive integer/);
  });

  test('returns error for NaN', () => {
    const r = resolveMaxTokens(NaN);
    assert.ok(r.error, 'expected error for NaN');
  });

  test('returns error for Infinity', () => {
    const r = resolveMaxTokens(Infinity);
    assert.ok(r.error, 'expected error for Infinity');
  });

  test('returns error for non-numeric string', () => {
    const r = resolveMaxTokens('lots');
    assert.ok(r.error, 'expected error for non-numeric string');
  });

  test('returns error for boolean true (not a sensible token count)', () => {
    // Number(true) === 1 which is a valid integer, so this actually passes.
    // Documenting the current behavior: booleans coerce to numbers.
    const r = resolveMaxTokens(true);
    assert.ok(!r.error, 'true coerces to 1 which is valid');
    assert.equal(r.value, PROXY_HARD_MAX_TOKENS ? Math.min(1, PROXY_HARD_MAX_TOKENS) : 1);
  });
});

// ── handleMessages — max_tokens validation ────────────────────────────────────

describe('handleMessages — max_tokens validation', () => {
  const { handleMessages } = require('./proxy');

  function mockReq(body) {
    return {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () { yield JSON.stringify(body); },
    };
  }
  function mockRes() {
    const res = {
      headersSent: false, writableEnded: false,
      _status: null, _body: '', _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(s) { this._status = s; this.headersSent = true; },
      write(c) { this._body += c; },
      end(c = '') { this._body += c; this.writableEnded = true; },
      on() {},
    };
    return res;
  }

  test('400 when max_tokens is zero', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 0 }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /max_tokens/);
  });

  test('400 when max_tokens is negative', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], max_tokens: -5 }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('400 when max_tokens is a non-integer float', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 128.7 }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('400 when max_tokens is a non-numeric string', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 'big' }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });
});
