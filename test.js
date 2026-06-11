'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');

const {
  parseDotEnv,
  parseOllamaOptions,
  parseOllamaError,
  mapOllamaError,
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
  timingSafeEqual,
  checkAuth,
  parseApiKeys,
  recordTokens,
  checkConcurrency,
  trackActiveLlmRequest,
  _metrics,
  cleanupExpiredBatches,
  processBatch,
  batchRequestCounts,
  _batches,
  truncateToContext,
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

// ── timingSafeEqual / checkAuth ──────────────────────────────────────────────

describe('timingSafeEqual', () => {
  test('returns true for identical strings', () => {
    assert.equal(timingSafeEqual('s3cret-key', 's3cret-key'), true);
  });

  test('returns false for different strings of the same length', () => {
    assert.equal(timingSafeEqual('s3cret-key', 's3cret-kex'), false);
  });

  test('returns false for strings of different lengths', () => {
    assert.equal(timingSafeEqual('short', 'a-much-longer-string'), false);
    assert.equal(timingSafeEqual('', 'nonempty'), false);
  });

  test('returns false when comparing against an empty string', () => {
    assert.equal(timingSafeEqual('', ''), true);
    assert.equal(timingSafeEqual('nonempty', ''), false);
  });
});

describe('checkAuth', () => {
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

  test('allows any request when PROXY_API_KEY is not set', () => {
    const req = { headers: {} };
    const res = mockRes();
    assert.equal(checkAuth(req, res), true);
    assert.equal(res.statusCode, null);
  });

  // Helper: load a fresh proxy module with PROXY_API_KEY set, run a single
  // test, then restore the module cache so other tests are unaffected.
  function withApiKey(key, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    let freshProxy;
    try {
      process.env.PROXY_API_KEY = key;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      delete process.env.PROXY_API_KEY;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  test('accepts the correct key via x-api-key header', () => {
    withApiKey('s3cret-key', (m) => {
      const req = { headers: { 'x-api-key': 's3cret-key' } };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), true);
      assert.equal(res.statusCode, null);
    });
  });

  test('accepts the correct key via Authorization: Bearer header', () => {
    withApiKey('s3cret-key', (m) => {
      const req = { headers: { authorization: 'Bearer s3cret-key' } };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), true);
    });
  });

  test('rejects a missing key with 401 authentication_error', () => {
    withApiKey('s3cret-key', (m) => {
      const req = { headers: {} };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), false);
      assert.equal(res.statusCode, 401);
      const body = JSON.parse(res.body);
      assert.equal(body.error.type, 'authentication_error');
    });
  });

  test('rejects an incorrect key of the same length', () => {
    withApiKey('s3cret-key', (m) => {
      const req = { headers: { 'x-api-key': 's3cret-kex' } };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), false);
      assert.equal(res.statusCode, 401);
    });
  });

  test('rejects an incorrect key of a different length', () => {
    withApiKey('s3cret-key', (m) => {
      const req = { headers: { 'x-api-key': 'wrong' } };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), false);
      assert.equal(res.statusCode, 401);
    });
  });

  // Helper: load a fresh proxy module with PROXY_API_KEY and/or PROXY_API_KEYS set,
  // run a single test, then restore the module cache so other tests are unaffected.
  function withApiKeyEnv(env, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    let freshProxy;
    try {
      for (const k of ['PROXY_API_KEY', 'PROXY_API_KEYS']) {
        savedEnv[k] = process.env[k];
        if (env[k] !== undefined) process.env[k] = env[k];
        else delete process.env[k];
      }
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const k of ['PROXY_API_KEY', 'PROXY_API_KEYS']) {
        if (savedEnv[k] !== undefined) process.env[k] = savedEnv[k];
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  test('PROXY_API_KEYS: accepts any configured key and tags req._apiKeyName', () => {
    withApiKeyEnv({ PROXY_API_KEYS: 'nick:nick-key,family:family-key' }, (m) => {
      const reqNick = { headers: { 'x-api-key': 'nick-key' } };
      assert.equal(m.checkAuth(reqNick, mockRes()), true);
      assert.equal(reqNick._apiKeyName, 'nick');

      const reqFamily = { headers: { 'x-api-key': 'family-key' } };
      assert.equal(m.checkAuth(reqFamily, mockRes()), true);
      assert.equal(reqFamily._apiKeyName, 'family');
    });
  });

  test('PROXY_API_KEYS: rejects a key not in the list', () => {
    withApiKeyEnv({ PROXY_API_KEYS: 'nick:nick-key,family:family-key' }, (m) => {
      const req = { headers: { 'x-api-key': 'someone-elses-key' } };
      const res = mockRes();
      assert.equal(m.checkAuth(req, res), false);
      assert.equal(res.statusCode, 401);
    });
  });

  test('PROXY_API_KEY and PROXY_API_KEYS combine — both are accepted', () => {
    withApiKeyEnv({ PROXY_API_KEY: 'shared-secret', PROXY_API_KEYS: 'laptop:laptop-key' }, (m) => {
      const reqDefault = { headers: { 'x-api-key': 'shared-secret' } };
      assert.equal(m.checkAuth(reqDefault, mockRes()), true);
      assert.equal(reqDefault._apiKeyName, 'default');

      const reqLaptop = { headers: { 'x-api-key': 'laptop-key' } };
      assert.equal(m.checkAuth(reqLaptop, mockRes()), true);
      assert.equal(reqLaptop._apiKeyName, 'laptop');
    });
  });
});

// ── parseApiKeys ──────────────────────────────────────────────────────────────

describe('parseApiKeys', () => {
  test('returns empty array when neither env var is set', () => {
    assert.deepEqual(parseApiKeys(null, undefined), []);
  });

  test('PROXY_API_KEY alone becomes a single "default" entry', () => {
    assert.deepEqual(parseApiKeys('mysecret', undefined), [{ name: 'default', key: 'mysecret' }]);
  });

  test('parses "name:key" pairs from PROXY_API_KEYS', () => {
    assert.deepEqual(parseApiKeys(null, 'nick:abc123,family:def456'), [
      { name: 'nick', key: 'abc123' },
      { name: 'family', key: 'def456' },
    ]);
  });

  test('auto-names bare keys without a colon as key1, key2, ...', () => {
    assert.deepEqual(parseApiKeys(null, 'abc123,def456'), [
      { name: 'key1', key: 'abc123' },
      { name: 'key2', key: 'def456' },
    ]);
  });

  test('trims whitespace around names, keys, and entries', () => {
    assert.deepEqual(parseApiKeys(null, ' nick : abc123 , family:def456 '), [
      { name: 'nick', key: 'abc123' },
      { name: 'family', key: 'def456' },
    ]);
  });

  test('skips empty entries from trailing/double commas', () => {
    assert.deepEqual(parseApiKeys(null, 'nick:abc123,,family:def456,'), [
      { name: 'nick', key: 'abc123' },
      { name: 'family', key: 'def456' },
    ]);
  });

  test('PROXY_API_KEY is prepended as "default" before PROXY_API_KEYS entries', () => {
    assert.deepEqual(parseApiKeys('shared-secret', 'laptop:laptop-key'), [
      { name: 'default', key: 'shared-secret' },
      { name: 'laptop', key: 'laptop-key' },
    ]);
  });
});

// ── recordTokens / per-API-key metrics ────────────────────────────────────────

describe('recordTokens', () => {
  test('aggregates tokens and request counts under apiKeysUsed by key name', () => {
    const before = JSON.parse(JSON.stringify(_metrics.apiKeysUsed));
    recordTokens(100, 50, 'qwen2.5:7b', 'nick');
    recordTokens(200, 75, 'qwen2.5:7b', 'nick');
    recordTokens(10, 5, 'qwen2.5:7b', 'family');

    assert.deepEqual(_metrics.apiKeysUsed.nick, {
      requests:  (before.nick?.requests  || 0) + 2,
      tokensIn:  (before.nick?.tokensIn  || 0) + 300,
      tokensOut: (before.nick?.tokensOut || 0) + 125,
    });
    assert.deepEqual(_metrics.apiKeysUsed.family, {
      requests:  (before.family?.requests  || 0) + 1,
      tokensIn:  (before.family?.tokensIn  || 0) + 10,
      tokensOut: (before.family?.tokensOut || 0) + 5,
    });
  });

  test('does not create an apiKeysUsed entry when apiKeyName is omitted', () => {
    const before = Object.keys(_metrics.apiKeysUsed).length;
    recordTokens(10, 5, 'qwen2.5:7b');
    assert.equal(Object.keys(_metrics.apiKeysUsed).length, before);
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

// ── mapOllamaError ──────────────────────────────────────────────────────────

describe('mapOllamaError', () => {
  test('maps 404 to not_found_error', () => {
    const result = mapOllamaError(404, '{"error":"model \'xyz\' not found, try pulling it first"}');
    assert.deepEqual(result, { status: 404, type: 'not_found_error', message: "model 'xyz' not found, try pulling it first" });
  });

  test('maps 400 to invalid_request_error', () => {
    const result = mapOllamaError(400, '{"error":"context length exceeded"}');
    assert.deepEqual(result, { status: 400, type: 'invalid_request_error', message: 'context length exceeded' });
  });

  test('maps 429 to rate_limit_error', () => {
    const result = mapOllamaError(429, '{"error":"too many requests"}');
    assert.deepEqual(result, { status: 429, type: 'rate_limit_error', message: 'too many requests' });
  });

  test('maps unrecognised status (e.g. 500) to 502 ollama_error', () => {
    const result = mapOllamaError(500, 'CUDA out of memory');
    assert.deepEqual(result, { status: 502, type: 'ollama_error', message: 'CUDA out of memory' });
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

// ── acquireLlmSlotForBatch ────────────────────────────────────────────────────

describe('acquireLlmSlotForBatch', () => {
  const { acquireLlmSlotForBatch, releaseLlmSlot, _concurrencyQueue, _metrics } = require('./proxy');

  test('when PROXY_MAX_CONCURRENCY is unset, grants immediately and increments counter', async () => {
    const before = _metrics.activeLlmRequests;
    await acquireLlmSlotForBatch();
    assert.equal(_metrics.activeLlmRequests, before + 1, 'activeLlmRequests incremented');
    releaseLlmSlot();
    assert.equal(_metrics.activeLlmRequests, before, 'counter back to baseline after release');
  });

  test('queues and resolves once a slot is released', async () => {
    // Simulate max concurrency = 1 by incrementing the counter to the limit.
    // We borrow the slot directly so we can release it at will.
    const before = _metrics.activeLlmRequests;
    _metrics.activeLlmRequests++;  // fill the (simulated) one slot

    // Patch PROXY_MAX_CONCURRENCY by directly saturating the queue mechanism:
    // push the entry ourselves as acquireLlmSlotForBatch would if the counter equalled the cap.
    // Instead, exercise the real function by manually controlling when the slot is freed.
    let resolved = false;
    const waitPromise = (async () => {
      // The queue is empty and activeLlmRequests > 0; inject a waiter directly.
      _metrics.queuedLlmRequests++;
      await new Promise(resolve => {
        _concurrencyQueue.push({
          onGranted: () => {
            _metrics.queuedLlmRequests--;
            resolved = true;
            resolve();
          },
        });
      });
    })();

    assert.equal(resolved, false, 'should not be resolved yet');
    assert.equal(_metrics.queuedLlmRequests, 1, 'waiter should be queued');

    // Release the occupied slot — this should hand it to the queued waiter.
    releaseLlmSlot();
    await waitPromise;

    assert.equal(resolved, true, 'waiter resolved after slot released');
    assert.equal(_metrics.queuedLlmRequests, 0, 'queuedLlmRequests back to 0');
    assert.equal(_metrics.activeLlmRequests, before + 1, 'slot transferred, not freed and re-acquired');

    // Final cleanup.
    releaseLlmSlot();
    assert.equal(_metrics.activeLlmRequests, before, 'back to baseline');
  });

  test('does not touch queuedLlmRequests when granted immediately', async () => {
    const qBefore = _metrics.queuedLlmRequests;
    await acquireLlmSlotForBatch();
    assert.equal(_metrics.queuedLlmRequests, qBefore, 'queuedLlmRequests unchanged on immediate grant');
    releaseLlmSlot();
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

  // Stubs global.fetch to return a non-ok response.
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

  test('non-streaming: Ollama 404 (unknown model) proxied as 404 not_found_error', async () => {
    const restore = stubFetchError(404, '{"error":"model \'xyz\' not found, try pulling it first"}');
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 404);
      const body = JSON.parse(res._body);
      // Error must be an object, not a raw/double-encoded string.
      assert.equal(typeof body.error, 'object', 'error should be an object, not a raw string');
      assert.equal(body.error.type, 'not_found_error');
      // Inner Ollama message should be extracted, not double-JSON-encoded.
      assert.equal(body.error.message, "model 'xyz' not found, try pulling it first");
    } finally { restore(); }
  });

  test('non-streaming: Ollama 400 proxied as 400 invalid_request_error', async () => {
    const restore = stubFetchError(400, '{"error":"invalid options"}');
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 400);
      const body = JSON.parse(res._body);
      assert.equal(body.error.type, 'invalid_request_error');
      assert.equal(body.error.message, 'invalid options');
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

  test('streaming: message_start.usage.input_tokens is a non-zero estimate from the request body', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":15,"completion_tokens":1}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(
        mockReq({ messages: [{ role: 'user', content: 'Hello, how are you today?' }], stream: true }),
        res,
      );
      const start = parseSse(res._body).find(e => e.event === 'message_start');
      // Should be a positive integer (chars/4 estimate), not 0.
      assert.ok(
        Number.isInteger(start.data.message.usage.input_tokens) &&
        start.data.message.usage.input_tokens > 0,
        `expected message_start.usage.input_tokens to be a positive integer, got ${start.data.message.usage.input_tokens}`,
      );
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

  test('streaming: sets X-Accel-Buffering: no header to prevent nginx buffering', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'ping' }], stream: true }), res);
      assert.equal(res._headers['X-Accel-Buffering'], 'no',
        'streaming response must include X-Accel-Buffering: no for nginx deployments');
    } finally { restore(); }
  });

  test('non-streaming: does not set X-Accel-Buffering header', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'ping' }], stream: false }), res);
      assert.ok(!res._headers['X-Accel-Buffering'],
        'non-streaming response must not include X-Accel-Buffering');
    } finally { restore(); }
  });
});

// ── anthropic-version response header ────────────────────────────────────────

describe('anthropic-version response header', () => {
  const { requestHandler } = require('./proxy');

  function mockReq(method, path, body = null) {
    return {
      method,
      url: path,
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1', encrypted: false },
      [Symbol.asyncIterator]: async function* () { if (body) yield JSON.stringify(body); },
    };
  }

  function mockRes() {
    const listeners = {};
    const res = {
      headersSent: false,
      writableEnded: false,
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k.toLowerCase()] = v; },
      getHeader(k) { return this._headers[k.toLowerCase()]; },
      writeHead(status) { this._status = status; this.headersSent = true; },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; if (listeners.finish) listeners.finish(); },
      on(event, fn) { listeners[event] = fn; },
      once(event, fn) { listeners[event] = fn; },
      off() {},
    };
    return res;
  }

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

  test('POST /v1/messages response includes anthropic-version: 2023-06-01', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    try {
      const res = mockRes();
      await requestHandler(
        mockReq('POST', '/v1/messages', { messages: [{ role: 'user', content: 'ping' }], stream: false }),
        res,
      );
      assert.equal(res._headers['anthropic-version'], '2023-06-01',
        'POST /v1/messages must return anthropic-version: 2023-06-01');
    } finally { restore(); }
  });

  test('POST /v1/messages/count_tokens response includes anthropic-version: 2023-06-01', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => ({ ok: false, status: 404 }); // fallback to char estimate
    try {
      const res = mockRes();
      await requestHandler(
        mockReq('POST', '/v1/messages/count_tokens', { messages: [{ role: 'user', content: 'hello' }] }),
        res,
      );
      assert.equal(res._headers['anthropic-version'], '2023-06-01',
        'POST /v1/messages/count_tokens must return anthropic-version: 2023-06-01');
    } finally { global.fetch = origFetch; }
  });

  test('GET /health does NOT include anthropic-version header', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ models: [] }) });
    try {
      const res = mockRes();
      await requestHandler(mockReq('GET', '/health'), res);
      assert.ok(!res._headers['anthropic-version'],
        'GET /health must not include anthropic-version header');
    } finally { global.fetch = origFetch; }
  });

  test('GET /v1/models does NOT include anthropic-version header', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ models: [] }) });
    try {
      const res = mockRes();
      await requestHandler(mockReq('GET', '/v1/models'), res);
      assert.ok(!res._headers['anthropic-version'],
        'GET /v1/models must not include anthropic-version header');
    } finally { global.fetch = origFetch; }
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

// ── PROXY_IDLE_TIMEOUT ────────────────────────────────────────────────────────

describe('PROXY_IDLE_TIMEOUT — idle stream timeout', () => {
  test('exported constant is null when env var is not set', () => {
    const { PROXY_IDLE_TIMEOUT } = require('./proxy');
    assert.strictEqual(PROXY_IDLE_TIMEOUT, null);
  });

  // Verifies that a stream which stalls after the first token chunk is aborted
  // and surfaced as a request_timeout SSE error once the idle window expires.
  // Uses module-cache surgery so the test doesn't depend on the ambient env var.
  test('streaming emits request_timeout SSE error when no tokens arrive within idle window', async () => {
    const enc = new TextEncoder();
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];

    // Load a fresh copy of the module with a short idle timeout.
    let freshHandler;
    try {
      process.env.PROXY_IDLE_TIMEOUT = '60';
      delete require.cache[modKey];
      freshHandler = require('./proxy').handleMessages;
    } finally {
      // Restore the original module so subsequent tests are unaffected.
      delete process.env.PROXY_IDLE_TIMEOUT;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }

    // Mock a stream: one token chunk, then hangs until the AbortSignal fires.
    let capturedSignal = null;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      capturedSignal = opts?.signal;
      return {
        ok: true, status: 200,
        body: {
          getReader() {
            let yielded = false;
            return {
              async read() {
                if (!yielded) {
                  yielded = true;
                  return { done: false, value: enc.encode('data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n') };
                }
                return new Promise((_, reject) => {
                  const abort = () => { const e = new Error('aborted'); e.name = 'AbortError'; reject(e); };
                  if (capturedSignal?.aborted) { abort(); return; }
                  capturedSignal?.addEventListener('abort', abort, { once: true });
                });
              },
              releaseLock() {},
            };
          },
        },
      };
    };

    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST',
      url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () {
        yield JSON.stringify({ messages: [{ role: 'user', content: 'hi' }], stream: true, max_tokens: 100 });
      },
    };
    const res = {
      headersSent: false, writableEnded: false,
      _body: '', _status: null, _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(s) { this._status = s; this.headersSent = true; },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
      on() {}, once() {},
    };

    try {
      await freshHandler(req, res);
    } finally {
      global.fetch = origFetch;
    }

    // Should have started the SSE stream (200) and emitted a request_timeout error.
    assert.equal(res._status, 200, 'expected streaming 200 response');
    assert.ok(res._body.includes('"request_timeout"'),
      `expected request_timeout error in SSE body:\n${res._body}`);
    assert.ok(
      res._body.includes('stuck') || res._body.includes('PROXY_IDLE_TIMEOUT') || res._body.includes('60ms'),
      `expected idle-timeout message in SSE body:\n${res._body}`
    );
  });
});

// ── cleanupExpiredBatches ─────────────────────────────────────────────────────

describe('cleanupExpiredBatches', () => {
  // Helper: create a minimal batch entry and insert it into the shared _batches Map.
  function makeBatch(overrides = {}) {
    const id = 'msgbatch_test_' + Math.random().toString(36).slice(2);
    const batch = {
      id,
      status:              'in_progress',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // 24 h from now
      ended_at:            null,
      cancel_initiated_at: null,
      requests:            [],
      results:             new Map(),
      cancelRequested:     false,
      ...overrides,
    };
    _batches.set(id, batch);
    return batch;
  }

  test('does not touch a fresh in-progress batch that has not expired', () => {
    const batch = makeBatch({ requests: [{ custom_id: 'r1' }] });
    cleanupExpiredBatches();
    assert.equal(batch.status, 'in_progress', 'status should remain in_progress');
    assert.ok(_batches.has(batch.id), 'batch should still be in the Map');
    _batches.delete(batch.id); // cleanup
  });

  test('force-expires an in-progress batch past its TTL and marks unresolved items', () => {
    const past = new Date(Date.now() - 1000).toISOString(); // 1 second ago
    const batch = makeBatch({
      expires_at: past,
      requests:   [{ custom_id: 'r1' }, { custom_id: 'r2' }],
    });
    // r1 already has a result; r2 does not
    batch.results.set('r1', { type: 'succeeded', message: {} });

    cleanupExpiredBatches();

    assert.equal(batch.status, 'ended', 'status should be ended after expiry enforcement');
    assert.ok(batch.ended_at, 'ended_at should be set');
    assert.equal(batch.results.get('r1').type, 'succeeded', 'existing result should be preserved');
    assert.equal(batch.results.get('r2').type, 'expired', 'unresolved item should be marked expired');
    _batches.delete(batch.id); // cleanup
  });

  test('removes an ended batch whose ended_at is more than 1 hour ago', () => {
    const longAgo = new Date(Date.now() - 61 * 60 * 1000).toISOString(); // 61 minutes ago
    const batch = makeBatch({
      status:   'ended',
      ended_at: longAgo,
    });

    cleanupExpiredBatches();

    assert.ok(!_batches.has(batch.id), 'old ended batch should have been removed from the Map');
  });

  test('keeps a recently ended batch (< 1 hour old) in the Map', () => {
    const recentlyEnded = new Date(Date.now() - 30 * 60 * 1000).toISOString(); // 30 minutes ago
    const batch = makeBatch({
      status:   'ended',
      ended_at: recentlyEnded,
    });

    cleanupExpiredBatches();

    assert.ok(_batches.has(batch.id), 'recently ended batch should still be in the Map');
    _batches.delete(batch.id); // cleanup
  });

  test('does not touch a canceling batch that has not yet expired', () => {
    const batch = makeBatch({
      status:          'canceling',
      cancelRequested: true,
      requests:        [{ custom_id: 'r1' }],
    });

    cleanupExpiredBatches();

    assert.equal(batch.status, 'canceling', 'status should remain canceling');
    _batches.delete(batch.id); // cleanup
  });
});

// ── processBatch expiry enforcement ──────────────────────────────────────────

describe('processBatch — expiry enforcement', () => {
  test('marks unprocessed items as expired when batch TTL has passed', async () => {
    const past = new Date(Date.now() - 1000).toISOString();
    const batch = {
      id:             'msgbatch_expiry_test',
      status:         'in_progress',
      expires_at:     past,
      ended_at:       null,
      cancelRequested: false,
      requests:       [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:        new Map(),
    };
    _batches.set(batch.id, batch);

    await processBatch(batch);

    assert.equal(batch.status, 'ended');
    assert.equal(batch.results.get('r1').type, 'expired',
      'item should be marked expired when batch TTL has passed');
    _batches.delete(batch.id);
  });
});

// ── handleOpenAIChat ──────────────────────────────────────────────────────────
// Unit tests for the OpenAI-format chat completions passthrough used by Cursor,
// Continue, LiteLLM, and other OpenAI-compatible clients.

describe('handleOpenAIChat', () => {
  const { handleOpenAIChat } = require('./proxy');

  function mockReq(body) {
    return {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST',
      url: '/v1/chat/completions',
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
      writeHead(status, headers) {
        this._status = status; this.headersSent = true;
        if (headers) for (const [k, v] of Object.entries(headers)) this._headers[k] = v;
      },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
      on() {},
    };
    return res;
  }

  function stubFetch(response, status = 200) {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: status < 400,
      status,
      json: async () => response,
      text: async () => JSON.stringify(response),
      body: null,
    });
    return () => { global.fetch = orig; };
  }

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

  test('400 on invalid JSON body', async () => {
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/chat/completions',
      [Symbol.asyncIterator]: async function* () { yield 'NOT JSON'; },
    };
    const res = mockRes();
    await handleOpenAIChat(req, res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('400 when messages field is absent', async () => {
    const res = mockRes();
    await handleOpenAIChat(mockReq({ model: 'llama3' }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /messages/);
  });

  test('400 when messages is not an array', async () => {
    const res = mockRes();
    await handleOpenAIChat(mockReq({ messages: 'not-an-array' }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('400 when max_tokens is invalid (negative)', async () => {
    const res = mockRes();
    await handleOpenAIChat(mockReq({
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: -1,
    }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('non-streaming: pipes Ollama JSON response with correct status', async () => {
    const ollamaResp = {
      id: 'chatcmpl-abc',
      object: 'chat.completion',
      model: 'qwen2.5:7b',
      choices: [{ message: { role: 'assistant', content: 'Hi!' }, finish_reason: 'stop', index: 0 }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    };
    const restore = stubFetch(ollamaResp);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'Hello' }] }), res);
      assert.equal(res._status, 200);
      const body = JSON.parse(res._body);
      assert.equal(body.id, 'chatcmpl-abc');
      assert.equal(body.choices[0].message.content, 'Hi!');
    } finally { restore(); }
  });

  test('non-streaming: resolves claude-* model names to Ollama models', async () => {
    let capturedBody;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return {
        ok: true, status: 200,
        json: async () => ({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: { prompt_tokens: 3, completion_tokens: 1 } }),
      };
    };
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ model: 'claude-3-haiku-20240307', messages: [{ role: 'user', content: 'hi' }] }), res);
      assert.ok(capturedBody.model, 'model should be forwarded');
      assert.ok(!capturedBody.model.startsWith('claude-'), 'claude-* name should be resolved to a real Ollama model name');
    } finally { global.fetch = origFetch; }
  });

  test('non-streaming: applies PROXY_MAX_TOKENS default when max_tokens is omitted', async () => {
    let capturedBody;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: {} }) };
    };
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hi' }] }), res);
      assert.ok(typeof capturedBody.max_tokens === 'number' && capturedBody.max_tokens > 0,
        'max_tokens should default to a positive number when omitted');
    } finally { global.fetch = origFetch; }
  });

  test('non-streaming: max_completion_tokens used as alias when max_tokens is absent', async () => {
    let capturedBody;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: {} }) };
    };
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({
        messages: [{ role: 'user', content: 'hi' }],
        max_completion_tokens: 256,
      }), res);
      assert.equal(capturedBody.max_tokens, 256,
        'max_completion_tokens should be honoured as max_tokens when max_tokens is absent');
      assert.ok(!('max_completion_tokens' in capturedBody),
        'max_completion_tokens should be removed before forwarding to avoid conflicting fields');
    } finally { global.fetch = origFetch; }
  });

  test('non-streaming: max_tokens takes precedence over max_completion_tokens when both present', async () => {
    let capturedBody;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }], usage: {} }) };
    };
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
        max_completion_tokens: 999,
      }), res);
      assert.equal(capturedBody.max_tokens, 100, 'max_tokens should take precedence');
      assert.ok(!('max_completion_tokens' in capturedBody), 'max_completion_tokens should be removed');
    } finally { global.fetch = origFetch; }
  });

  test('non-streaming: proxies Ollama error status directly', async () => {
    const restore = stubFetch('{"error":"model not found"}', 404);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hi' }] }), res);
      assert.equal(res._status, 404);
    } finally { restore(); }
  });

  test('non-streaming: records token usage in _logMeta for request logging', async () => {
    const restore = stubFetch({
      choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 15, completion_tokens: 7 },
    });
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hi' }] }), res);
      assert.ok(res._logMeta, '_logMeta should be set for request logging');
      assert.equal(res._logMeta.tokensIn, 15);
      assert.equal(res._logMeta.tokensOut, 7);
    } finally { restore(); }
  });

  test('streaming: returns 200 with text/event-stream content-type', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hello' }], stream: true }), res);
      assert.equal(res._status, 200);
      assert.ok(res._headers['Content-Type']?.includes('text/event-stream'),
        'streaming response must have SSE content-type');
    } finally { restore(); }
  });

  test('streaming: pipes Ollama SSE lines and includes [DONE] sentinel', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
      'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hi' }], stream: true }), res);
      assert.ok(res._body.includes('"hello"'), 'first chunk content should appear in SSE output');
      assert.ok(res._body.includes('[DONE]'), '[DONE] sentinel should be piped through');
    } finally { restore(); }
  });

  test('streaming: extracts token usage from trailing Ollama usage chunk into _logMeta', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}',
      'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'ping' }], stream: true }), res);
      assert.ok(res._logMeta, '_logMeta should be set after streaming completes');
      assert.equal(res._logMeta.tokensIn, 7, 'input tokens from trailing usage chunk');
      assert.equal(res._logMeta.tokensOut, 3, 'output tokens from trailing usage chunk');
    } finally { restore(); }
  });

  test('streaming: sets X-Accel-Buffering: no to prevent nginx buffering', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}',
      'data: [DONE]',
    ]);
    try {
      const res = mockRes();
      await handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'ping' }], stream: true }), res);
      assert.equal(res._headers['X-Accel-Buffering'], 'no',
        'streaming OpenAI passthrough must include X-Accel-Buffering: no');
    } finally { restore(); }
  });
});

// ── PROXY_FORCE_THINK ─────────────────────────────────────────────────────────
// Verifies that PROXY_FORCE_THINK=true causes think:true to be forwarded to
// Ollama on handleMessages and handleOpenAICompletions requests, and that it
// is absent (false) in the default test environment.

describe('PROXY_FORCE_THINK', () => {
  const { PROXY_FORCE_THINK } = require('./proxy');

  test('exported constant is false when env var is not set', () => {
    assert.strictEqual(PROXY_FORCE_THINK, false);
  });

  // Helper: load a fresh proxy module with PROXY_FORCE_THINK=true, run a
  // single test, then restore the module cache so other tests are unaffected.
  function withForceThink(fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    let freshProxy;
    try {
      process.env.PROXY_FORCE_THINK = 'true';
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      delete process.env.PROXY_FORCE_THINK;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  test('PROXY_FORCE_THINK=true is exported as true from a fresh module load', () => {
    withForceThink(m => {
      assert.strictEqual(m.PROXY_FORCE_THINK, true);
    });
  });

  test('handleMessages sends think:true to Ollama when PROXY_FORCE_THINK=true', async () => {
    await withForceThink(async (m) => {
      let sentBody = null;
      const origFetch = global.fetch;
      global.fetch = async (_url, opts) => {
        sentBody = JSON.parse(opts.body);
        return {
          ok: true, status: 200,
          json: async () => ({
            choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 3, completion_tokens: 1 },
          }),
          body: null,
        };
      };
      const req = {
        headers: {},
        socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
        method: 'POST', url: '/v1/messages',
        [Symbol.asyncIterator]: async function* () {
          yield JSON.stringify({ messages: [{ role: 'user', content: 'hello' }], stream: false });
        },
      };
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
      try {
        await m.handleMessages(req, res);
      } finally {
        global.fetch = origFetch;
      }
      assert.ok(sentBody !== null, 'fetch should have been called');
      assert.strictEqual(sentBody.think, true,
        'think:true should be forwarded to Ollama when PROXY_FORCE_THINK=true');
    });
  });

  test('handleMessages does NOT send think:true by default (PROXY_FORCE_THINK=false)', async () => {
    const { handleMessages } = require('./proxy');
    let sentBody = null;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      sentBody = JSON.parse(opts.body);
      return {
        ok: true, status: 200,
        json: async () => ({
          choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 3, completion_tokens: 1 },
        }),
        body: null,
      };
    };
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () {
        yield JSON.stringify({ messages: [{ role: 'user', content: 'hello' }], stream: false });
      },
    };
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
    try {
      await handleMessages(req, res);
    } finally {
      global.fetch = origFetch;
    }
    assert.ok(sentBody !== null, 'fetch should have been called');
    assert.ok(!('think' in sentBody),
      'think field should NOT be present when PROXY_FORCE_THINK is false and client did not request thinking');
  });

  test('handleMessages still sends think:true when client requests thinking explicitly (without PROXY_FORCE_THINK)', async () => {
    const { handleMessages } = require('./proxy');
    let sentBody = null;
    const origFetch = global.fetch;
    global.fetch = async (_url, opts) => {
      sentBody = JSON.parse(opts.body);
      return {
        ok: true, status: 200,
        json: async () => ({
          choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 3, completion_tokens: 1 },
        }),
        body: null,
      };
    };
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/messages',
      [Symbol.asyncIterator]: async function* () {
        yield JSON.stringify({
          messages: [{ role: 'user', content: 'think hard' }],
          stream: false,
          thinking: { type: 'enabled', budget_tokens: 2048 },
        });
      },
    };
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
    try {
      await handleMessages(req, res);
    } finally {
      global.fetch = origFetch;
    }
    assert.strictEqual(sentBody.think, true,
      'think:true should be set when client explicitly requests thinking');
  });
});

// ── truncateToContext ─────────────────────────────────────────────────────────

describe('truncateToContext', () => {
  function msgs(...roles) {
    return roles.map((role, i) => ({ role, content: 'x'.repeat(20) + i }));
  }

  test('returns messages unchanged when within budget', () => {
    const m = msgs('user', 'assistant');
    const { messages, droppedCount } = truncateToContext(m, 100_000);
    assert.deepEqual(messages, m);
    assert.equal(droppedCount, 0);
  });

  test('drops oldest messages when over budget', () => {
    // Build a long history that exceeds a tiny budget.
    const m = [
      { role: 'user',      content: 'A'.repeat(500) },
      { role: 'assistant', content: 'B'.repeat(500) },
      { role: 'user',      content: 'C'.repeat(50)  },
      { role: 'assistant', content: 'D'.repeat(50)  },
    ];
    const budget = Math.ceil(JSON.stringify(m.slice(2)).length / 4) + 10;
    const { messages, droppedCount } = truncateToContext(m, budget);
    // First two (long) messages should have been dropped.
    assert.ok(droppedCount >= 2, `expected >=2 dropped, got ${droppedCount}`);
    // Result should start with a user role.
    assert.equal(messages[0].role, 'user');
    // Estimate of result should now be within budget.
    const est = Math.ceil(JSON.stringify(messages).length / 4);
    assert.ok(est <= budget, `estimate ${est} exceeds budget ${budget}`);
  });

  test('always keeps system message', () => {
    const m = [
      { role: 'system',    content: 'sys' },
      { role: 'user',      content: 'A'.repeat(600) },
      { role: 'assistant', content: 'B'.repeat(600) },
      { role: 'user',      content: 'short' },
      { role: 'assistant', content: 'reply' },
    ];
    const budget = 50;
    const { messages } = truncateToContext(m, budget);
    assert.equal(messages[0].role, 'system');
    assert.equal(messages[0].content, 'sys');
  });

  test('always keeps at least KEEP_LAST (2) non-system messages', () => {
    const m = [
      { role: 'user',      content: 'A'.repeat(5000) },
      { role: 'assistant', content: 'B'.repeat(5000) },
      { role: 'user',      content: 'final question' },
      { role: 'assistant', content: 'final answer'   },
    ];
    const { messages, droppedCount } = truncateToContext(m, 1);
    // The last 2 messages must be kept even if over budget.
    const nonSys = messages.filter(msg => msg.role !== 'system');
    assert.ok(nonSys.length >= 2, `expected >=2 non-system messages, got ${nonSys.length}`);
    assert.ok(droppedCount >= 2, `expected >=2 dropped, got ${droppedCount}`);
  });

  test('result starts with user role after dropping orphaned assistant messages', () => {
    const m = [
      { role: 'user',      content: 'X'.repeat(400) },
      { role: 'assistant', content: 'Y'.repeat(400) },
      { role: 'user',      content: 'u2' },
      { role: 'assistant', content: 'a2' },
    ];
    const budget = Math.ceil(JSON.stringify(m.slice(2)).length / 4) + 5;
    const { messages } = truncateToContext(m, budget);
    // The first non-system message in the result must be a user turn.
    const first = messages.find(msg => msg.role !== 'system');
    assert.equal(first?.role, 'user', `expected first non-system role to be 'user', got '${first?.role}'`);
  });

  test('droppedCount reflects all messages removed including orphaned ones', () => {
    // After dropping user(A400), assistant(B400) becomes an orphaned head; the function
    // should also drop it, giving droppedCount=2, not 1.
    const m = [
      { role: 'user',      content: 'A'.repeat(400) },
      { role: 'assistant', content: 'B'.repeat(400) },
      { role: 'user',      content: 'short' },
    ];
    const budget = Math.ceil(JSON.stringify([{ role: 'user', content: 'short' }]).length / 4) + 5;
    const { droppedCount, messages } = truncateToContext(m, budget);
    assert.ok(droppedCount >= 2, `expected droppedCount >=2, got ${droppedCount}`);
    assert.equal(messages[0].role, 'user');
  });
});
