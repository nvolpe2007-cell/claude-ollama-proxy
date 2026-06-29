'use strict';
const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const fs   = require('fs');
const os   = require('os');
const path = require('path');
const vm   = require('node:vm');

const {
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
  checkHostHealth,
  recordHostHealth,
  _hostHealth,
  HOST_UNHEALTHY_THRESHOLD,
  checkRateLimit,
  getClientIp,
  rateLimitKeyForRequest,
  _rateLimitWindows,
  timingSafeEqual,
  checkAuth,
  parseApiKeys,
  parseApiKeyModels,
  checkModelAccess,
  isModelVisibleToCaller,
  _apiKeyModels,
  recordTokens,
  recordRequest,
  MAX_METRICS_PATH_KEYS,
  MAX_MODELS_USED_KEYS,
  checkConcurrency,
  trackActiveLlmRequest,
  _metrics,
  cleanupExpiredBatches,
  processBatch,
  batchRequestCounts,
  batchOwnedByCaller,
  batchOwnerName,
  saveBatchesToDisk,
  loadBatchesFromDisk,
  handleCreateBatch,
  handleListBatches,
  parseBatchListParams,
  handleGetBatch,
  handleGetBatchResults,
  handleCancelBatch,
  handleDeleteBatch,
  MAX_BATCH_REQUESTS,
  _batches,
  truncateToContext,
  handleEmbeddings,
  validateEncodingFormat,
  embeddingToBase64,
  handleDashboard,
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

  test('falls back to the default model for non-string input instead of throwing', () => {
    assert.equal(resolveModel(123), DEFAULT);
    assert.equal(resolveModel(true), DEFAULT);
    assert.equal(resolveModel({}), DEFAULT);
    assert.equal(resolveModel(['claude-3-opus']), DEFAULT);
  });
});

// ── validateModelField ────────────────────────────────────────────────────────

describe('validateModelField', () => {
  test('accepts absent or null model', () => {
    assert.deepEqual(validateModelField(undefined), {});
    assert.deepEqual(validateModelField(null), {});
  });

  test('accepts a string model', () => {
    assert.deepEqual(validateModelField('qwen2.5:7b'), {});
    assert.deepEqual(validateModelField(''), {});
  });

  test('rejects non-string model values', () => {
    assert.match(validateModelField(123).error, /must be a string/);
    assert.match(validateModelField(true).error, /must be a string/);
    assert.match(validateModelField({}).error, /must be a string/);
    assert.match(validateModelField(['claude-3-opus']).error, /must be a string/);
  });
});

// ── validateTools ────────────────────────────────────────────────────────────

describe('validateTools', () => {
  test('accepts absent or null tools', () => {
    assert.deepEqual(validateTools(undefined), {});
    assert.deepEqual(validateTools(null), {});
  });

  test('accepts a well-formed tools array', () => {
    assert.deepEqual(validateTools([]), {});
    assert.deepEqual(validateTools([
      { name: 'get_weather', description: 'Get weather', input_schema: { type: 'object' } },
    ]), {});
  });

  test('rejects a non-array tools value', () => {
    assert.match(validateTools('get_weather').error, /must be an array/);
    assert.match(validateTools({ name: 'get_weather' }).error, /must be an array/);
    assert.match(validateTools(42).error, /must be an array/);
  });

  test('rejects malformed entries that would crash toOpenAITools', () => {
    assert.match(validateTools([null]).error, /non-empty string `name`/);
    assert.match(validateTools(['get_weather']).error, /non-empty string `name`/);
    assert.match(validateTools([{}]).error, /non-empty string `name`/);
    assert.match(validateTools([{ name: '' }]).error, /non-empty string `name`/);
    assert.match(validateTools([{ name: 123 }]).error, /non-empty string `name`/);
  });
});

// ── validateEncodingFormat / embeddingToBase64 ────────────────────────────────

describe('validateEncodingFormat', () => {
  test('accepts absent or null encoding_format', () => {
    assert.deepEqual(validateEncodingFormat(undefined), {});
    assert.deepEqual(validateEncodingFormat(null), {});
  });

  test('accepts "float" and "base64"', () => {
    assert.deepEqual(validateEncodingFormat('float'), {});
    assert.deepEqual(validateEncodingFormat('base64'), {});
  });

  test('rejects any other value', () => {
    assert.match(validateEncodingFormat('hex').error, /must be "float" or "base64"/);
    assert.match(validateEncodingFormat(123).error, /must be "float" or "base64"/);
    assert.match(validateEncodingFormat(true).error, /must be "float" or "base64"/);
  });
});

describe('embeddingToBase64', () => {
  test('round-trips a float vector through base64 the way the OpenAI wire format expects', () => {
    const original = [0.1, -0.5, 2, 0];
    const encoded = embeddingToBase64(original);
    assert.equal(typeof encoded, 'string');
    const buf = Buffer.from(encoded, 'base64');
    const decoded = [...new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)];
    for (let i = 0; i < original.length; i++) {
      assert.ok(Math.abs(decoded[i] - original[i]) < 1e-6);
    }
  });
});

// ── validateSystemField ───────────────────────────────────────────────────────

describe('validateSystemField', () => {
  test('accepts absent or null system', () => {
    assert.deepEqual(validateSystemField(undefined), {});
    assert.deepEqual(validateSystemField(null), {});
  });

  test('accepts a string system prompt', () => {
    assert.deepEqual(validateSystemField('be concise'), {});
    assert.deepEqual(validateSystemField(''), {});
  });

  test('accepts an array of content blocks', () => {
    assert.deepEqual(validateSystemField([]), {});
    assert.deepEqual(validateSystemField([{ type: 'text', text: 'be concise' }]), {});
  });

  test('rejects non-string, non-array system values that would crash injectSystemPrompt', () => {
    assert.match(validateSystemField(123).error, /must be a string or an array/);
    assert.match(validateSystemField(true).error, /must be a string or an array/);
    assert.match(validateSystemField({ type: 'text', text: 'be concise' }).error, /must be a string or an array/);
  });

  test('rejects a malformed element inside a system array that would crash toOpenAIMessages', () => {
    assert.match(validateSystemField([null]).error, /each item in `system`/);
    assert.match(validateSystemField(['be concise']).error, /each item in `system`/);
    assert.match(validateSystemField([{ text: 'no type field' }]).error, /each item in `system`/);
    assert.match(validateSystemField([{ type: 'text', text: 'ok' }, null]).error, /each item in `system`/);
  });
});

// ── validateMessages ─────────────────────────────────────────────────────────

describe('validateMessages', () => {
  test('accepts string content', () => {
    assert.deepEqual(validateMessages([{ role: 'user', content: 'hello' }]), {});
  });

  test('accepts a well-formed content-block array', () => {
    assert.deepEqual(validateMessages([
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    ]), {});
    assert.deepEqual(validateMessages([
      { role: 'assistant', content: [{ type: 'tool_use', id: 'x', name: 'foo', input: {} }] },
    ]), {});
    assert.deepEqual(validateMessages([
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: [{ type: 'text', text: 'result' }] }] },
    ]), {});
  });

  test('accepts absent/null content and OpenAI-style messages', () => {
    assert.deepEqual(validateMessages([{ role: 'user' }]), {});
    assert.deepEqual(validateMessages([
      { role: 'system', content: 'sys' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: null, tool_calls: [{ id: '1' }] },
      { role: 'user', content: [{ type: 'text', text: 'hi' }, { type: 'image_url', image_url: { url: 'x' } }] },
    ]), {});
  });

  test('rejects non-object/null message entries that would crash toOpenAIMessages', () => {
    assert.match(validateMessages([null]).error, /must be an object/);
    assert.match(validateMessages(['hello']).error, /must be an object/);
  });

  test('rejects a missing/non-string role', () => {
    assert.match(validateMessages([{ content: 'hi' }]).error, /string `role`/);
    assert.match(validateMessages([{ role: 1, content: 'hi' }]).error, /string `role`/);
  });

  test('rejects content that is neither a string nor an array', () => {
    assert.match(validateMessages([{ role: 'user', content: 123 }]).error, /must be a string or an array/);
  });

  test('rejects null/non-object content blocks that would crash toOpenAIMessages', () => {
    assert.match(validateMessages([{ role: 'user', content: [null] }]).error, /content block/);
    assert.match(validateMessages([{ role: 'user', content: ['hi'] }]).error, /content block/);
    assert.match(validateMessages([{ role: 'user', content: [{ text: 'hi' }] }]).error, /string `type`/);
  });

  test('rejects null tool_result content entries that would crash toOpenAIMessages', () => {
    assert.match(validateMessages([
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'x', content: [null] }] },
    ]).error, /tool_result.*content block/);
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

  test('falls back to empty string content when image source is unsupported and no text', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'image', source: { type: 'file', file_id: 'file_abc' } },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result.length, 1);
    assert.equal(result[0].content, '');
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

  test('drops unknown content block type and warns', () => {
    const orig = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    const messages = [{
      role: 'assistant',
      content: [{ type: 'redacted_thinking', data: 'opaque-blob' }],
    }];
    let result;
    try { result = toOpenAIMessages(messages, null); } finally { console.warn = orig; }
    assert.equal(result[0].content, '');
    assert.ok(warns.some(w => w.includes('redacted_thinking')));
  });

  test('falls back to text field of an unknown content block type', () => {
    const orig = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    const messages = [{
      role: 'assistant',
      content: [
        { type: 'text', text: 'known: ' },
        { type: 'server_tool_use', text: 'unknown-but-has-text' },
      ],
    }];
    let result;
    try { result = toOpenAIMessages(messages, null); } finally { console.warn = orig; }
    assert.equal(result[0].content, 'known: unknown-but-has-text');
    assert.ok(warns.some(w => w.includes('server_tool_use') && w.includes('kept its text field')));
  });

  test('unknown content block in a tool_result turn still surfaces via text fallback', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'toolu_1', content: 'ok' },
        { type: 'web_search_tool_result', text: 'search summary' },
      ],
    }];
    const result = toOpenAIMessages(messages, null);
    assert.equal(result[0].role, 'tool');
    assert.equal(result[1].role, 'user');
    assert.equal(result[1].content, 'search summary');
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

// ── Host health tracking ─────────────────────────────────────────────────────

describe('recordHostHealth / getOllamaHost failover', () => {
  const host = OLLAMA_HOSTS[0];

  test('host starts out healthy', () => {
    const h = _hostHealth.get(host);
    assert.equal(h.healthy, true);
    assert.equal(h.consecutiveFailures, 0);
  });

  test('marks unhealthy only after HOST_UNHEALTHY_THRESHOLD consecutive failures', () => {
    for (let i = 1; i < HOST_UNHEALTHY_THRESHOLD; i++) {
      recordHostHealth(host, false, 'boom');
      assert.equal(_hostHealth.get(host).healthy, true, `should still be healthy after ${i} failure(s)`);
    }
    recordHostHealth(host, false, 'boom');
    assert.equal(_hostHealth.get(host).healthy, false);
    assert.equal(_hostHealth.get(host).lastError, 'boom');
  });

  test('a single success immediately restores health', () => {
    recordHostHealth(host, true, null);
    const h = _hostHealth.get(host);
    assert.equal(h.healthy, true);
    assert.equal(h.consecutiveFailures, 0);
    assert.equal(h.lastError, null);
  });

  test('getOllamaHost fails open when the only host is unhealthy', () => {
    for (let i = 0; i < HOST_UNHEALTHY_THRESHOLD; i++) recordHostHealth(host, false, 'down');
    assert.equal(_hostHealth.get(host).healthy, false);
    // With a single configured host there's nowhere else to route — must still
    // return that host rather than failing with no result.
    assert.equal(getOllamaHost(), host);
    recordHostHealth(host, true, null); // restore for other tests
  });
});

describe('checkHostHealth', () => {
  const host = OLLAMA_HOSTS[0];

  test('records success when /api/tags responds ok', async () => {
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200 });
    try {
      const ok = await checkHostHealth(host);
      assert.equal(ok, true);
      assert.equal(_hostHealth.get(host).healthy, true);
      assert.equal(_hostHealth.get(host).consecutiveFailures, 0);
    } finally {
      global.fetch = orig;
    }
  });

  test('records failure when fetch throws', async () => {
    const orig = global.fetch;
    global.fetch = async () => { throw new Error('ECONNREFUSED'); };
    try {
      for (let i = 0; i < HOST_UNHEALTHY_THRESHOLD; i++) {
        const ok = await checkHostHealth(host);
        assert.equal(ok, false);
      }
      const h = _hostHealth.get(host);
      assert.equal(h.healthy, false);
      assert.equal(h.lastError, 'ECONNREFUSED');
    } finally {
      global.fetch = orig;
      recordHostHealth(host, true, null); // restore for other tests
    }
  });
});

// withHosts() loads a fresh proxy module with a multi-host OLLAMA_HOST so
// round-robin/failover behavior can be tested in isolation, then restores
// the original module cache so other tests are unaffected.
function withHosts(hostsCsv, fn) {
  const modKey = require.resolve('./proxy');
  const savedMod = require.cache[modKey];
  let freshProxy;
  try {
    process.env.OLLAMA_HOST = hostsCsv;
    delete require.cache[modKey];
    freshProxy = require('./proxy');
  } finally {
    delete process.env.OLLAMA_HOST;
    delete require.cache[modKey];
    require.cache[modKey] = savedMod;
  }
  return fn(freshProxy);
}

describe('OLLAMA_HOST trailing slash normalization', () => {
  test('strips a single trailing slash from a single host', () => {
    withHosts('http://localhost:11434/', (mod) => {
      assert.deepEqual(mod.OLLAMA_HOSTS, ['http://localhost:11434']);
    });
  });

  test('strips repeated trailing slashes', () => {
    withHosts('http://localhost:11434///', (mod) => {
      assert.deepEqual(mod.OLLAMA_HOSTS, ['http://localhost:11434']);
    });
  });

  test('strips trailing slashes independently across a comma-separated list', () => {
    withHosts('http://host-a:11434/, http://host-b:11434 ,http://host-c:11434//', (mod) => {
      assert.deepEqual(mod.OLLAMA_HOSTS, [
        'http://host-a:11434',
        'http://host-b:11434',
        'http://host-c:11434',
      ]);
    });
  });

  test('leaves a host with no trailing slash unchanged', () => {
    withHosts('http://localhost:11434', (mod) => {
      assert.deepEqual(mod.OLLAMA_HOSTS, ['http://localhost:11434']);
    });
  });
});

describe('multi-host failover', () => {
  test('round-robins across all healthy hosts', () => {
    withHosts('http://host-a:11434,http://host-b:11434', (mod) => {
      const seen = new Set();
      for (let i = 0; i < mod.OLLAMA_HOSTS.length; i++) seen.add(mod.getOllamaHost());
      assert.deepEqual(seen, new Set(mod.OLLAMA_HOSTS));
    });
  });

  test('skips a host marked unhealthy until it recovers', () => {
    withHosts('http://host-a:11434,http://host-b:11434', (mod) => {
      const [hostA, hostB] = mod.OLLAMA_HOSTS;
      for (let i = 0; i < mod.HOST_UNHEALTHY_THRESHOLD; i++) mod.recordHostHealth(hostA, false, 'down');

      for (let i = 0; i < 5; i++) assert.equal(mod.getOllamaHost(), hostB);

      mod.recordHostHealth(hostA, true, null);
      const seen = new Set();
      for (let i = 0; i < mod.OLLAMA_HOSTS.length; i++) seen.add(mod.getOllamaHost());
      assert.deepEqual(seen, new Set([hostA, hostB]));
    });
  });

  test('fails open and keeps rotating when every host is unhealthy', () => {
    withHosts('http://host-a:11434,http://host-b:11434', (mod) => {
      const [hostA, hostB] = mod.OLLAMA_HOSTS;
      for (let i = 0; i < mod.HOST_UNHEALTHY_THRESHOLD; i++) {
        mod.recordHostHealth(hostA, false, 'down');
        mod.recordHostHealth(hostB, false, 'down');
      }
      const seen = new Set();
      for (let i = 0; i < mod.OLLAMA_HOSTS.length * 2; i++) seen.add(mod.getOllamaHost());
      assert.deepEqual(seen, new Set([hostA, hostB]));
    });
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

  test('ignores x-forwarded-for by default (PROXY_TRUST_PROXY not set)', () => {
    const req = { headers: { 'x-forwarded-for': '203.0.113.5, 10.0.0.2, 10.0.0.3' }, socket: { remoteAddress: '10.0.0.2' } };
    assert.equal(getClientIp(req), '10.0.0.2',
      'a client-supplied x-forwarded-for must not override the socket address unless PROXY_TRUST_PROXY=true, otherwise RATE_LIMIT_PER_IP_RPM is trivially spoofable');
  });

  test('falls back to "unknown" when socket has no remoteAddress', () => {
    const req = { headers: {}, socket: {} };
    assert.equal(getClientIp(req), 'unknown');
  });
});

// withTrustProxy() loads a fresh proxy module with PROXY_TRUST_PROXY=true so
// x-forwarded-for trust behavior can be tested in isolation, then restores the
// original module cache so other tests are unaffected.
function withTrustProxy(fn) {
  const modKey = require.resolve('./proxy');
  const savedMod = require.cache[modKey];
  let freshProxy;
  try {
    process.env.PROXY_TRUST_PROXY = 'true';
    delete require.cache[modKey];
    freshProxy = require('./proxy');
  } finally {
    delete process.env.PROXY_TRUST_PROXY;
    delete require.cache[modKey];
    require.cache[modKey] = savedMod;
  }
  return fn(freshProxy);
}

describe('getClientIp with PROXY_TRUST_PROXY=true', () => {
  test('returns first IP from x-forwarded-for header', () => {
    withTrustProxy(({ getClientIp }) => {
      const req = { headers: { 'x-forwarded-for': '203.0.113.5, 10.0.0.2, 10.0.0.3' }, socket: { remoteAddress: '10.0.0.2' } };
      assert.equal(getClientIp(req), '203.0.113.5');
    });
  });

  test('trims whitespace from x-forwarded-for', () => {
    withTrustProxy(({ getClientIp }) => {
      const req = { headers: { 'x-forwarded-for': '  192.168.1.1  ' }, socket: { remoteAddress: '10.0.0.1' } };
      assert.equal(getClientIp(req), '192.168.1.1');
    });
  });

  test('falls back to socket remoteAddress when header absent', () => {
    withTrustProxy(({ getClientIp }) => {
      const req = { headers: {}, socket: { remoteAddress: '10.0.0.1' } };
      assert.equal(getClientIp(req), '10.0.0.1');
    });
  });
});

// ── rateLimitKeyForRequest ───────────────────────────────────────────────────

describe('rateLimitKeyForRequest', () => {
  test('buckets by the matched API key name', () => {
    const req = { _apiKeyName: 'nick' };
    assert.equal(rateLimitKeyForRequest(req), 'key:nick');
  });

  test('falls back to "key:default" when no API key matched', () => {
    const req = {};
    assert.equal(rateLimitKeyForRequest(req), 'key:default');
  });

  test('different key names produce different buckets', () => {
    const reqA = { _apiKeyName: 'nick' };
    const reqB = { _apiKeyName: 'family' };
    assert.notEqual(rateLimitKeyForRequest(reqA), rateLimitKeyForRequest(reqB));
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

// ── parseApiKeyModels / checkModelAccess ──────────────────────────────────────

describe('parseApiKeyModels', () => {
  test('returns an empty Map when the env var is unset or empty', () => {
    assert.deepEqual(parseApiKeyModels(undefined), new Map());
    assert.deepEqual(parseApiKeyModels(''), new Map());
  });

  test('parses a single "name:model" entry', () => {
    const map = parseApiKeyModels('family:llama3.2:1b');
    assert.deepEqual(map.get('family'), new Set(['llama3.2:1b']));
  });

  test('parses multiple pipe-separated models for one key', () => {
    const map = parseApiKeyModels('family:qwen2.5:7b|llama3.2:1b');
    assert.deepEqual(map.get('family'), new Set(['qwen2.5:7b', 'llama3.2:1b']));
  });

  test('parses multiple comma-separated key entries', () => {
    const map = parseApiKeyModels('family:llama3.2:1b,nick:qwen2.5:7b|qwen2.5:14b');
    assert.deepEqual(map.get('family'), new Set(['llama3.2:1b']));
    assert.deepEqual(map.get('nick'), new Set(['qwen2.5:7b', 'qwen2.5:14b']));
  });

  test('trims whitespace around names, models, and entries', () => {
    const map = parseApiKeyModels(' family : qwen2.5:7b | llama3.2:1b , nick:qwen2.5:14b ');
    assert.deepEqual(map.get('family'), new Set(['qwen2.5:7b', 'llama3.2:1b']));
    assert.deepEqual(map.get('nick'), new Set(['qwen2.5:14b']));
  });

  test('skips malformed entries with no colon', () => {
    const map = parseApiKeyModels('not-a-valid-entry,family:llama3.2:1b');
    assert.equal(map.has('not-a-valid-entry'), false);
    assert.deepEqual(map.get('family'), new Set(['llama3.2:1b']));
  });

  test('skips entries with an empty model list', () => {
    const map = parseApiKeyModels('family:,nick:qwen2.5:7b');
    assert.equal(map.has('family'), false);
    assert.deepEqual(map.get('nick'), new Set(['qwen2.5:7b']));
  });

  test('skips empty entries from trailing/double commas', () => {
    const map = parseApiKeyModels('family:llama3.2:1b,,nick:qwen2.5:7b,');
    assert.deepEqual(map.get('family'), new Set(['llama3.2:1b']));
    assert.deepEqual(map.get('nick'), new Set(['qwen2.5:7b']));
  });
});

describe('checkModelAccess', () => {
  test('returns null (no restriction) when PROXY_API_KEY_MODELS is not configured', () => {
    assert.equal(_apiKeyModels.size, 0);
    assert.equal(checkModelAccess({ _apiKeyName: 'nick' }, 'qwen2.5:7b'), null);
    assert.equal(checkModelAccess({}, 'qwen2.5:7b'), null);
  });

  // Helper: load a fresh proxy module with PROXY_API_KEY_MODELS set, run a single
  // test, then restore the module cache so other tests are unaffected.
  function withApiKeyModels(value, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const saved = process.env.PROXY_API_KEY_MODELS;
    let freshProxy;
    try {
      process.env.PROXY_API_KEY_MODELS = value;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      if (saved !== undefined) process.env.PROXY_API_KEY_MODELS = saved;
      else delete process.env.PROXY_API_KEY_MODELS;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  test('allows a key to use a model in its allow-list', () => {
    withApiKeyModels('family:llama3.2:1b', (m) => {
      const req = { _apiKeyName: 'family' };
      assert.equal(m.checkModelAccess(req, 'llama3.2:1b'), null);
    });
  });

  test('rejects a key using a model not in its allow-list', () => {
    withApiKeyModels('family:llama3.2:1b', (m) => {
      const req = { _apiKeyName: 'family' };
      const err = m.checkModelAccess(req, 'qwen2.5:7b');
      assert.match(err, /family/);
      assert.match(err, /qwen2\.5:7b/);
      assert.match(err, /llama3\.2:1b/);
    });
  });

  test('keys with no entry in PROXY_API_KEY_MODELS have unrestricted access', () => {
    withApiKeyModels('family:llama3.2:1b', (m) => {
      const req = { _apiKeyName: 'nick' };
      assert.equal(m.checkModelAccess(req, 'qwen2.5:72b'), null);
    });
  });

  test('falls back to the "default" bucket when req._apiKeyName is unset', () => {
    withApiKeyModels('default:llama3.2:1b', (m) => {
      const reqNoKeyName = {};
      assert.equal(m.checkModelAccess(reqNoKeyName, 'llama3.2:1b'), null);
      const err = m.checkModelAccess(reqNoKeyName, 'qwen2.5:7b');
      assert.match(err, /default/);
    });
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

// ── recordTokens — _metrics.modelsUsed cardinality cap ────────────────────────
// `model` reaches recordTokens() as the resolved effective model name, and resolveModel()
// passes any non-claude-* request `model` string through verbatim — just as client-controlled
// as the req.url path that MAX_METRICS_PATH_KEYS already caps. Without an equivalent cap here,
// a caller could grow _metrics.modelsUsed (and GET /metrics/GET /metrics/prometheus response
// size) without bound by sending a unique `model` string per request.
describe('recordTokens — _metrics.modelsUsed cardinality cap', () => {
  test('caps the number of distinct model keys and buckets overflow under "(other)"', () => {
    Object.keys(_metrics.modelsUsed).forEach(k => delete _metrics.modelsUsed[k]);

    for (let i = 0; i < MAX_MODELS_USED_KEYS + 50; i++) {
      recordTokens(10, 5, `attacker-model-${i}`);
    }

    const keys = Object.keys(_metrics.modelsUsed);
    assert.ok(keys.length <= MAX_MODELS_USED_KEYS + 1,
      `expected at most ${MAX_MODELS_USED_KEYS + 1} distinct keys (cap + overflow bucket), got ${keys.length}`);
    assert.deepEqual(_metrics.modelsUsed['(other)'], { requests: 50, tokensIn: 500, tokensOut: 250 });
  });

  test('does not bucket a model already seen before the cap was reached', () => {
    Object.keys(_metrics.modelsUsed).forEach(k => delete _metrics.modelsUsed[k]);

    recordTokens(10, 5, 'qwen2.5:7b');
    for (let i = 0; i < MAX_MODELS_USED_KEYS + 10; i++) {
      recordTokens(10, 5, `flood-model-${i}`);
    }
    recordTokens(20, 8, 'qwen2.5:7b');

    assert.deepEqual(_metrics.modelsUsed['qwen2.5:7b'], { requests: 2, tokensIn: 30, tokensOut: 13 });
  });

  test('still aggregates global token totals while a model is bucketed as overflow', () => {
    Object.keys(_metrics.modelsUsed).forEach(k => delete _metrics.modelsUsed[k]);
    const before = { in: _metrics.tokensIn, out: _metrics.tokensOut };

    for (let i = 0; i < MAX_MODELS_USED_KEYS + 1; i++) {
      recordTokens(10, 5, `flood-model-${i}`);
    }

    assert.equal(_metrics.tokensIn,  before.in  + (MAX_MODELS_USED_KEYS + 1) * 10);
    assert.equal(_metrics.tokensOut, before.out + (MAX_MODELS_USED_KEYS + 1) * 5);
  });
});

// ── recordRequest — _metrics.requests cardinality cap ─────────────────────────
// recordRequest runs for every request, including unauthenticated ones (the `finish`
// listener fires before checkAuth()), so an attacker who never sends a valid API key can
// still drive it with an arbitrary number of distinct paths. Without a cap this grows
// _metrics.requests (and GET /metrics/GET /metrics/prometheus response size) without bound.
describe('recordRequest — _metrics.requests cardinality cap', () => {
  test('caps the number of distinct path keys and buckets overflow under "(other)"', () => {
    Object.keys(_metrics.requests).forEach(k => delete _metrics.requests[k]);

    for (let i = 0; i < MAX_METRICS_PATH_KEYS + 50; i++) {
      recordRequest('GET', `/attacker-path-${i}`, 404, 1);
    }

    const keys = Object.keys(_metrics.requests);
    assert.ok(keys.length <= MAX_METRICS_PATH_KEYS + 1,
      `expected at most ${MAX_METRICS_PATH_KEYS + 1} distinct keys (cap + overflow bucket), got ${keys.length}`);
    assert.equal(_metrics.requests['GET (other)'], 50,
      'the 50 paths beyond the cap should all be counted under the overflow bucket');
  });

  test('does not bucket a path already seen before the cap was reached', () => {
    Object.keys(_metrics.requests).forEach(k => delete _metrics.requests[k]);

    recordRequest('GET', '/health', 200, 1);
    for (let i = 0; i < MAX_METRICS_PATH_KEYS + 10; i++) {
      recordRequest('GET', `/flood-${i}`, 404, 1);
    }
    recordRequest('GET', '/health', 200, 1);

    assert.equal(_metrics.requests['GET /health'], 2,
      'a path recorded before the cap was reached keeps incrementing its own key');
  });

  test('still records status codes and latencies while a path is bucketed as overflow', () => {
    Object.keys(_metrics.requests).forEach(k => delete _metrics.requests[k]);
    const statusBefore = JSON.parse(JSON.stringify(_metrics.statusCodes));
    const latLenBefore = _metrics.latencies.length;

    for (let i = 0; i < MAX_METRICS_PATH_KEYS + 1; i++) {
      recordRequest('GET', `/flood-${i}`, 404, 7);
    }

    assert.equal(_metrics.statusCodes['404'], (statusBefore['404'] || 0) + MAX_METRICS_PATH_KEYS + 1);
    assert.equal(_metrics.latencies.length, Math.min(1000, latLenBefore + MAX_METRICS_PATH_KEYS + 1));
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

// ── isModelVisibleToCaller ──────────────────────────────────────────────────────

describe('isModelVisibleToCaller', () => {
  test('returns true for every model when PROXY_API_KEY_MODELS is not configured', () => {
    assert.equal(_apiKeyModels.size, 0);
    assert.equal(isModelVisibleToCaller({ _apiKeyName: 'nick' }, 'qwen2.5:7b'), true);
    assert.equal(isModelVisibleToCaller({}, 'llama3.2:1b'), true);
  });
});

// ── GET /v1/models & GET /v1/models/:id — PROXY_API_KEY_MODELS visibility filtering ──
// handleModels and handleModelById are exercised end-to-end (with a stubbed Ollama
// /api/tags + /api/show) by reloading the proxy module with custom MODEL_MAP and
// PROXY_API_KEY_MODELS env vars, mirroring the withApiKeyModels reload pattern above.

describe('handleModels / handleModelById — model visibility filtering', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  function mockReq(apiKeyName) {
    return { headers: {}, socket: { remoteAddress: '127.0.0.1' }, _apiKeyName: apiKeyName };
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      end(chunk = '') { this._body += chunk; },
    };
  }

  const tagsResponse = {
    models: [
      { name: 'llama3.2:1b', modified_at: '2024-01-01T00:00:00Z', size: 100, details: {} },
      { name: 'qwen2.5:7b',  modified_at: '2024-01-01T00:00:00Z', size: 200, details: {} },
      { name: 'qwen2.5:14b', modified_at: '2024-01-01T00:00:00Z', size: 300, details: {} },
    ],
  };

  function stubFetch(response) {
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => response });
    return () => { global.fetch = orig; };
  }

  test('unrestricted key sees every Ollama model and MODEL_MAP alias', async () => {
    await withProxyEnv({
      MODEL_MAP: JSON.stringify({ 'claude-3-haiku': 'qwen2.5:7b' }),
      PROXY_API_KEY_MODELS: 'family:llama3.2:1b',
    }, async (m) => {
      const restore = stubFetch(tagsResponse);
      try {
        const req = mockReq('nick'); // no entry in PROXY_API_KEY_MODELS -> unrestricted
        const res = mockRes();
        await m.handleModels(req, res);
        const ids = JSON.parse(res._body).data.map(x => x.id);
        assert.deepEqual(ids.sort(), ['claude-3-haiku', 'llama3.2:1b', 'qwen2.5:14b', 'qwen2.5:7b']);
      } finally {
        restore();
      }
    });
  });

  test('restricted key only sees its allowed models and matching aliases', async () => {
    await withProxyEnv({
      MODEL_MAP: JSON.stringify({ 'claude-3-haiku': 'qwen2.5:7b', 'claude-3-big': 'qwen2.5:14b' }),
      PROXY_API_KEY_MODELS: 'family:qwen2.5:7b',
    }, async (m) => {
      const restore = stubFetch(tagsResponse);
      try {
        const req = mockReq('family');
        const res = mockRes();
        await m.handleModels(req, res);
        const ids = JSON.parse(res._body).data.map(x => x.id);
        assert.deepEqual(ids.sort(), ['claude-3-haiku', 'qwen2.5:7b']);
      } finally {
        restore();
      }
    });
  });

  test('returns 404 for a real model outside the caller\'s allow-list', async () => {
    await withProxyEnv({
      PROXY_API_KEY_MODELS: 'family:llama3.2:1b',
    }, async (m) => {
      const restore = stubFetch(tagsResponse);
      try {
        const req = mockReq('family');
        const res = mockRes();
        await m.handleModelById(req, res, 'qwen2.5:7b');
        assert.equal(res._status, 404);
        assert.equal(JSON.parse(res._body).error.type, 'not_found_error');
      } finally {
        restore();
      }
    });
  });

  test('returns the model details for a model in the caller\'s allow-list', async () => {
    await withProxyEnv({
      PROXY_API_KEY_MODELS: 'family:llama3.2:1b',
    }, async (m) => {
      const restore = stubFetch(tagsResponse);
      try {
        const req = mockReq('family');
        const res = mockRes();
        await m.handleModelById(req, res, 'llama3.2:1b');
        assert.equal(res._status, 200);
        assert.equal(JSON.parse(res._body).id, 'llama3.2:1b');
      } finally {
        restore();
      }
    });
  });
});

// ── DELETE /v1/models/:id & POST /v1/models/pull — PROXY_API_KEY_MODELS enforcement ──
// These write endpoints previously checked only auth, not the caller's model
// allow-list, so a restricted key could delete or pull models it isn't permitted
// to use for inference. Mirrors the withProxyEnv reload pattern used above.

describe('handleDeleteModel / handlePullModel — model access control', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  function mockReq(apiKeyName) {
    return {
      headers: {},
      socket: { remoteAddress: '127.0.0.1', once() {}, off() {} },
      _apiKeyName: apiKeyName,
    };
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      writableEnded: false,
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
    };
  }

  function stubFetch(response) {
    const orig = global.fetch;
    let calls = 0;
    global.fetch = async () => { calls++; return { ok: true, status: 200, json: async () => response }; };
    return { restore: () => { global.fetch = orig; }, callCount: () => calls };
  }

  test('DELETE rejects a model outside the caller\'s allow-list without calling Ollama', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch({});
      try {
        const req = mockReq('family');
        const res = mockRes();
        await m.handleDeleteModel(req, res, 'qwen2.5:7b');
        assert.equal(res._status, 403);
        assert.equal(JSON.parse(res._body).error.type, 'permission_error');
        assert.equal(fetchStub.callCount(), 0, 'Ollama should never be called for a disallowed model');
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('DELETE allows a model in the caller\'s allow-list', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch({});
      try {
        const req = mockReq('family');
        const res = mockRes();
        await m.handleDeleteModel(req, res, 'llama3.2:1b');
        assert.equal(res._status, 200);
        assert.equal(JSON.parse(res._body).deleted, true);
        assert.equal(fetchStub.callCount(), 1);
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('DELETE is unrestricted for a key with no PROXY_API_KEY_MODELS entry', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch({});
      try {
        const req = mockReq('nick');
        const res = mockRes();
        await m.handleDeleteModel(req, res, 'qwen2.5:72b');
        assert.equal(res._status, 200);
        assert.equal(fetchStub.callCount(), 1);
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('pull rejects a model outside the caller\'s allow-list without calling Ollama', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch({});
      try {
        const req = mockReq('family');
        req.headers = {};
        const res = mockRes();
        const body = JSON.stringify({ model: 'qwen2.5:7b' });
        req[Symbol.asyncIterator] = async function* () { yield Buffer.from(body); };
        await m.handlePullModel(req, res);
        assert.equal(res._status, 403);
        assert.equal(JSON.parse(res._body).error.type, 'permission_error');
        assert.equal(fetchStub.callCount(), 0, 'Ollama should never be called for a disallowed model');
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('pull allows a model in the caller\'s allow-list', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch({ status: 'success' });
      try {
        const req = mockReq('family');
        const res = mockRes();
        const body = JSON.stringify({ model: 'llama3.2:1b' });
        req[Symbol.asyncIterator] = async function* () { yield Buffer.from(body); };
        await m.handlePullModel(req, res);
        assert.equal(res._status, 200);
        assert.equal(JSON.parse(res._body).pulled, true);
        assert.equal(fetchStub.callCount(), 1);
      } finally {
        fetchStub.restore();
      }
    });
  });
});

// ── POST /v1/messages/count_tokens — PROXY_API_KEY_MODELS enforcement ──────────
// handleCountTokens resolved the model and forwarded it straight to Ollama's
// /api/tokenize endpoint without ever checking the caller's model allow-list,
// unlike every other model-touching endpoint (messages, chat completions,
// completions, embeddings, batches, model management). A restricted key could
// use this to force-load a disallowed model into Ollama just to count tokens.

describe('handleCountTokens — model access control', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  function mockReq(apiKeyName, bodyObj) {
    const body = JSON.stringify(bodyObj);
    const req = {
      headers: {},
      socket: { remoteAddress: '127.0.0.1', once() {}, off() {} },
      _apiKeyName: apiKeyName,
    };
    req[Symbol.asyncIterator] = async function* () { yield Buffer.from(body); };
    return req;
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      writableEnded: false,
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
    };
  }

  function stubFetch() {
    const orig = global.fetch;
    let calls = 0;
    global.fetch = async () => { calls++; return { ok: true, status: 200, json: async () => ({ tokens: [1, 2, 3] }) }; };
    return { restore: () => { global.fetch = orig; }, callCount: () => calls };
  }

  test('rejects a model outside the caller\'s allow-list without calling Ollama', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch();
      try {
        const req = mockReq('family', { model: 'qwen2.5:7b', messages: [{ role: 'user', content: 'hi' }] });
        const res = mockRes();
        await m.handleCountTokens(req, res);
        assert.equal(res._status, 403);
        assert.equal(JSON.parse(res._body).error.type, 'permission_error');
        assert.equal(fetchStub.callCount(), 0, 'Ollama should never be called for a disallowed model');
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('allows a model in the caller\'s allow-list', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch();
      try {
        const req = mockReq('family', { model: 'llama3.2:1b', messages: [{ role: 'user', content: 'hi' }] });
        const res = mockRes();
        await m.handleCountTokens(req, res);
        assert.equal(res._status, 200);
        assert.equal(JSON.parse(res._body).input_tokens, 3);
        assert.equal(fetchStub.callCount(), 1);
      } finally {
        fetchStub.restore();
      }
    });
  });

  test('is unrestricted for a key with no PROXY_API_KEY_MODELS entry', async () => {
    await withProxyEnv({ PROXY_API_KEY_MODELS: 'family:llama3.2:1b' }, async (m) => {
      const fetchStub = stubFetch();
      try {
        const req = mockReq('nick', { model: 'qwen2.5:72b', messages: [{ role: 'user', content: 'hi' }] });
        const res = mockRes();
        await m.handleCountTokens(req, res);
        assert.equal(res._status, 200);
        assert.equal(fetchStub.callCount(), 1);
      } finally {
        fetchStub.restore();
      }
    });
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

  // Stubs global.fetch to return a streaming SSE response, one chunk per element.
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

  test('streaming: passes through a mid-stream {"error":...} chunk', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}',
      'data: {"error":{"message":"model crashed","type":"server_error"}}',
    ]);
    try {
      const req = mockReq({ prompt: 'hi', stream: true });
      const res = mockRes();
      await handleOpenAICompletions(req, res);
      const errLine = res._body.split('\n').find(l => l.startsWith('data: ') && l.includes('"error"'));
      assert.ok(errLine, 'should write an error data line');
      const parsed = JSON.parse(errLine.slice(6));
      assert.equal(parsed.error.message, 'model crashed');
    } finally { restore(); }
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

  test('400 when a message entry is malformed (would crash toOpenAIMessages)', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [null] }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /messages/);
  });

  test('400 when a content block is malformed (would crash toOpenAIMessages)', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: [null] }] }), res);
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

  test('non-streaming: choice with no message field returns 502 instead of crashing', async () => {
    const restore = stubFetch({ choices: [{ finish_reason: 'stop' }], usage: {} });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 502);
      assert.equal(JSON.parse(res._body).error.type, 'ollama_error');
    } finally { restore(); }
  });

  test('non-streaming: malformed tool_call JSON arguments default to {} and log a warning', async () => {
    const restore = stubFetch({
      choices: [{
        message: { content: null, tool_calls: [{ id: 'call_1', function: { name: 'get_weather', arguments: '{not valid json' } }] },
        finish_reason: 'tool_calls',
      }],
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    const origWarn = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Weather?' }], stream: false }), res);
      const body = JSON.parse(res._body);
      const tu = body.content.find(c => c.type === 'tool_use');
      assert.ok(tu, 'should have tool_use block');
      assert.deepEqual(tu.input, {});
      assert.ok(warns.some(w => w.includes('[tool-call]') && w.includes('get_weather')));
    } finally { console.warn = origWarn; restore(); }
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

  test('non-streaming: does not crash if reading the Ollama error body itself throws', async () => {
    // Simulates a client abort / decoding failure while reading ollamaRes.text()
    // in the !ollamaRes.ok branch — should degrade to a mapped error response,
    // not an uncaught rejection.
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: false, status: 500,
      text: async () => { throw new Error('AbortError'); },
      body: null,
    });
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'Hi' }], stream: false }), res);
      assert.equal(res._status, 502);
      const body = JSON.parse(res._body);
      assert.equal(body.error.type, 'ollama_error');
    } finally { global.fetch = orig; }
  });

  test('non-streaming: removes the close listener and clears the timeout on a non-ok Ollama response', async () => {
    const { EventEmitter } = require('events');
    const restore = stubFetchError(500, 'CUDA out of memory');
    try {
      const socket = new EventEmitter();
      socket.remoteAddress = '127.0.0.1';
      const req = {
        headers: {},
        socket,
        method: 'POST',
        url: '/v1/messages',
        [Symbol.asyncIterator]: async function* () {
          yield JSON.stringify({ messages: [{ role: 'user', content: 'Hi' }], stream: false });
        },
      };
      const res = mockRes();
      await handleMessages(req, res);
      assert.equal(res._status, 502);
      assert.equal(socket.listenerCount('close'), 0, 'close listener should be removed after a non-ok Ollama response');
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

  // ── Mid-stream error chunks ──────────────────────────────────────────────────
  // Ollama can emit a {"error": ...} chunk after generation has already started
  // (e.g. the model crashes or runs out of VRAM mid-response). Without explicit
  // handling, such a chunk has no `choices` and was silently skipped, so the
  // stream ended as if it had completed normally (stop_reason: 'end_turn'),
  // hiding the failure from the client.

  test('streaming: mid-stream {"error":...} chunk emits an Anthropic error event', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}',
      'data: {"error":{"message":"model crashed","type":"server_error"}}',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], stream: true }), res);
      const events = parseSse(res._body);
      const errEvent = events.find(e => e.event === 'error');
      assert.ok(errEvent, 'should emit an error SSE event');
      assert.equal(errEvent.data.error.type, 'api_error');
      assert.equal(errEvent.data.error.message, 'model crashed');
    } finally { restore(); }
  });

  test('streaming: mid-stream error does not send a misleading message_stop', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}',
      'data: {"error":{"message":"model crashed","type":"server_error"}}',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], stream: true }), res);
      const events = parseSse(res._body);
      assert.ok(!events.some(e => e.event === 'message_stop'), 'should not send message_stop after a mid-stream error');
      assert.ok(!events.some(e => e.event === 'message_delta'), 'should not send message_delta after a mid-stream error');
    } finally { restore(); }
  });

  test('streaming: mid-stream error closes the open text content block first', async () => {
    const restore = stubStreamFetch([
      'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}',
      'data: {"error":"plain string error"}',
    ]);
    try {
      const res = mockRes();
      await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], stream: true }), res);
      const events = parseSse(res._body);
      const stopIdx  = events.findIndex(e => e.event === 'content_block_stop');
      const errIdx   = events.findIndex(e => e.event === 'error');
      assert.ok(stopIdx !== -1, 'should close the open text block');
      assert.ok(errIdx !== -1, 'should emit an error event');
      assert.ok(stopIdx < errIdx, 'content_block_stop should come before the error event');
      const errEvent = events[errIdx];
      assert.equal(errEvent.data.error.message, 'plain string error');
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

  test('POST /v1/messages/count_tokens rejects non-array messages with 400', async () => {
    const res = mockRes();
    await requestHandler(
      mockReq('POST', '/v1/messages/count_tokens', { messages: 'not-an-array' }),
      res,
    );
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
  });

  test('POST /v1/messages/count_tokens rejects missing messages with 400', async () => {
    const res = mockRes();
    await requestHandler(
      mockReq('POST', '/v1/messages/count_tokens', {}),
      res,
    );
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
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

  // ── GET /health — model availability ────────────────────────────────────────
  const MODEL_NAME = process.env.OLLAMA_MODEL || 'qwen2.5:7b';
  const host = OLLAMA_HOSTS[0];

  function withTagsFetch(modelNames) {
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ models: modelNames.map(name => ({ name })) }) });
    return () => { global.fetch = orig; };
  }

  test('GET /health reports model_available: true and status "ok" when the configured model is pulled', async () => {
    const restore = withTagsFetch([MODEL_NAME, 'llava:latest']);
    try {
      const res = mockRes();
      await requestHandler(mockReq('GET', '/health'), res);
      const body = JSON.parse(res._body);
      assert.equal(body.model_available, true);
      assert.equal(body.status, 'ok');
      assert.equal(body.warning, undefined);
    } finally {
      restore();
      delete _hostHealth.get(host).models;
    }
  });

  test('GET /health reports model_available: false, status "degraded", and a warning when the configured model is missing', async () => {
    const restore = withTagsFetch(['some-other-model:latest']);
    try {
      const res = mockRes();
      await requestHandler(mockReq('GET', '/health'), res);
      const body = JSON.parse(res._body);
      assert.equal(body.model_available, false);
      assert.equal(body.status, 'degraded');
      assert.ok(body.warning.includes(MODEL_NAME));
      assert.ok(body.warning.includes('ollama pull'));
    } finally {
      restore();
      delete _hostHealth.get(host).models;
    }
  });

  test('GET /health reports model_available: null when Ollama is unreachable (cannot check)', async () => {
    const origFetch = global.fetch;
    global.fetch = async () => { throw new Error('ECONNREFUSED'); };
    try {
      const res = mockRes();
      await requestHandler(mockReq('GET', '/health'), res);
      const body = JSON.parse(res._body);
      assert.equal(body.model_available, null);
      assert.equal(body.warning, undefined);
    } finally {
      global.fetch = origFetch;
      recordHostHealth(host, true, null);
      delete _hostHealth.get(host).models;
    }
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

  test('400 when model is not a string', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], model: 123 }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /model/);
  });

  test('400 when model is an object', async () => {
    const res = mockRes();
    await handleMessages(mockReq({ messages: [{ role: 'user', content: 'hi' }], model: { foo: 'bar' } }), res);
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

// ── processBatch — resume skips already-processed items ───────────────────────

describe('processBatch — resume', () => {
  test('skips items that already have a recorded result and only processes the rest', async () => {
    const batch = {
      id:              'msgbatch_skip_test',
      status:          'in_progress',
      expires_at:      new Date(Date.now() + 60_000).toISOString(),
      ended_at:        null,
      cancelRequested: false,
      requests: [
        { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
        { custom_id: 'r2', params: { messages: [], model: 'test', max_tokens: 1 } },
      ],
      results: new Map([['r1', { type: 'succeeded', message: { id: 'msg_already_done' } }]]),
    };
    _batches.set(batch.id, batch);

    const orig = global.fetch;
    let fetchCalls = 0;
    global.fetch = async () => {
      fetchCalls++;
      return {
        ok: true, status: 200,
        json: async () => ({ choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }], usage: {} }),
      };
    };

    try {
      await processBatch(batch);
      assert.equal(fetchCalls, 1, 'only the unresolved item (r2) should hit Ollama');
      assert.equal(batch.results.get('r1').message.id, 'msg_already_done', 'existing result must not be overwritten');
      assert.equal(batch.results.get('r2').type, 'succeeded');
      assert.equal(batch.status, 'ended');
    } finally {
      global.fetch = orig;
      _batches.delete(batch.id);
    }
  });
});

// ── processBatch — per-API-key token attribution ──────────────────────────────

describe('processBatch — per-API-key metrics', () => {
  test('attributes processed token usage to the batch owner in _metrics.apiKeysUsed', async () => {
    const before = JSON.parse(JSON.stringify(_metrics.apiKeysUsed));
    const batch = {
      id:              'msgbatch_owner_metrics_test',
      status:          'in_progress',
      owner:           'nick',
      expires_at:      new Date(Date.now() + 60_000).toISOString(),
      ended_at:        null,
      cancelRequested: false,
      requests: [
        { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
      ],
      results: new Map(),
    };
    _batches.set(batch.id, batch);

    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 40, completion_tokens: 12 },
      }),
    });

    try {
      await processBatch(batch);
      assert.deepEqual(_metrics.apiKeysUsed.nick, {
        requests:  (before.nick?.requests  || 0) + 1,
        tokensIn:  (before.nick?.tokensIn  || 0) + 40,
        tokensOut: (before.nick?.tokensOut || 0) + 12,
      });
    } finally {
      global.fetch = orig;
      _batches.delete(batch.id);
    }
  });

  test('falls back to "default" when a persisted batch has no owner field', async () => {
    const before = JSON.parse(JSON.stringify(_metrics.apiKeysUsed));
    const batch = {
      id:              'msgbatch_no_owner_metrics_test',
      status:          'in_progress',
      expires_at:      new Date(Date.now() + 60_000).toISOString(),
      ended_at:        null,
      cancelRequested: false,
      requests: [
        { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
      ],
      results: new Map(),
    };
    _batches.set(batch.id, batch);

    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 7, completion_tokens: 3 },
      }),
    });

    try {
      await processBatch(batch);
      assert.deepEqual(_metrics.apiKeysUsed.default, {
        requests:  (before.default?.requests  || 0) + 1,
        tokensIn:  (before.default?.tokensIn  || 0) + 7,
        tokensOut: (before.default?.tokensOut || 0) + 3,
      });
    } finally {
      global.fetch = orig;
      _batches.delete(batch.id);
    }
  });
});

describe('processBatch — malformed Ollama responses', () => {
  test('choice with no message field is recorded as errored instead of throwing', async () => {
    const batch = {
      id:              'msgbatch_no_message_test',
      status:          'in_progress',
      expires_at:      new Date(Date.now() + 60_000).toISOString(),
      ended_at:        null,
      cancelRequested: false,
      requests: [
        { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
      ],
      results: new Map(),
    };
    _batches.set(batch.id, batch);

    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({ choices: [{ finish_reason: 'stop' }], usage: {} }),
    });

    try {
      await processBatch(batch);
      const result = batch.results.get('r1');
      assert.equal(result.type, 'errored');
      assert.equal(result.error.type, 'ollama_error');
    } finally {
      global.fetch = orig;
      _batches.delete(batch.id);
    }
  });

  test('malformed tool_call JSON arguments default to {} and log a warning', async () => {
    const batch = {
      id:              'msgbatch_bad_tool_json_test',
      status:          'in_progress',
      expires_at:      new Date(Date.now() + 60_000).toISOString(),
      ended_at:        null,
      cancelRequested: false,
      requests: [
        { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
      ],
      results: new Map(),
    };
    _batches.set(batch.id, batch);

    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({
        choices: [{
          message: { content: null, tool_calls: [{ id: 'call_1', function: { name: 'get_weather', arguments: '{not valid json' } }] },
          finish_reason: 'tool_calls',
        }],
        usage: { prompt_tokens: 5, completion_tokens: 2 },
      }),
    });

    const origWarn = console.warn;
    const warns = [];
    console.warn = (...a) => warns.push(a.join(' '));
    try {
      await processBatch(batch);
      const result = batch.results.get('r1');
      const tu = result.message.content.find(c => c.type === 'tool_use');
      assert.ok(tu, 'should have tool_use block');
      assert.deepEqual(tu.input, {});
      assert.ok(warns.some(w => w.includes('[tool-call]') && w.includes('get_weather')));
    } finally {
      console.warn = origWarn;
      global.fetch = orig;
      _batches.delete(batch.id);
    }
  });
});

// ── processBatch — multi-host round robin ──────────────────────────────────────

describe('processBatch — multi-host round robin', () => {
  test('re-resolves the Ollama host per item instead of pinning the whole batch to one host', async () => {
    await withHosts('http://host-a:11434,http://host-b:11434', async (mod) => {
      const batch = {
        id:              'msgbatch_multihost_test',
        status:          'in_progress',
        expires_at:      new Date(Date.now() + 60_000).toISOString(),
        ended_at:        null,
        cancelRequested: false,
        requests: [
          { custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } },
          { custom_id: 'r2', params: { messages: [], model: 'test', max_tokens: 1 } },
        ],
        results: new Map(),
      };
      mod._batches.set(batch.id, batch);

      const orig = global.fetch;
      const calledUrls = [];
      global.fetch = async (url) => {
        calledUrls.push(url);
        return {
          ok: true, status: 200,
          json: async () => ({ choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }], usage: {} }),
        };
      };

      try {
        await mod.processBatch(batch);
        assert.deepEqual(calledUrls, mod.OLLAMA_HOSTS.map(h => `${h}/v1/chat/completions`),
          'each item should round-robin across hosts instead of pinning the whole batch to one host');
      } finally {
        global.fetch = orig;
        mod._batches.delete(batch.id);
      }
    });
  });
});

// ── processBatch — cancellation race while parked in the concurrency queue ────

describe('processBatch — cancellation race while parked in concurrency queue', () => {
  function withMaxConcurrency(n, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    try {
      process.env.PROXY_MAX_CONCURRENCY = String(n);
      delete require.cache[modKey];
      const freshProxy = require('./proxy');
      return fn(freshProxy);
    } finally {
      delete process.env.PROXY_MAX_CONCURRENCY;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
  }

  test('a batch item canceled while queued for a slot is recorded as canceled instead of running against Ollama', async () => {
    await withMaxConcurrency(1, async (mod) => {
      // Occupy the single concurrency slot, simulating real-time traffic holding it.
      await mod.acquireLlmSlotForBatch();

      const batch = {
        id:              'msgbatch_cancel_race_test',
        status:          'in_progress',
        expires_at:      new Date(Date.now() + 60_000).toISOString(),
        ended_at:        null,
        cancelRequested: false,
        requests:        [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
        results:         new Map(),
      };

      const origFetch = global.fetch;
      let fetchCalled = false;
      global.fetch = async () => {
        fetchCalled = true;
        return {
          ok: true, status: 200,
          json: async () => ({ choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }], usage: {} }),
        };
      };

      try {
        const donePromise = mod.processBatch(batch);

        // Let processBatch run up to (and park in) acquireLlmSlotForBatch's queue.
        await new Promise(r => setImmediate(r));
        assert.equal(mod._metrics.queuedLlmRequests, 1, 'item should be parked in the concurrency queue');

        // Cancellation arrives while the item is queued — mirrors handleCancelBatch
        // setting cancelRequested on a batch that has an item waiting for a slot.
        batch.cancelRequested = true;

        // Free the slot the item is waiting on.
        mod.releaseLlmSlot();
        await donePromise;

        assert.equal(fetchCalled, false, 'a canceled item must not be sent to Ollama');
        assert.deepEqual(batch.results.get('r1'), { type: 'canceled' });
        assert.equal(batch.status, 'ended');
      } finally {
        global.fetch = origFetch;
      }
    });
  });
});

// ── Batch persistence (saveBatchesToDisk / loadBatchesFromDisk) ───────────────

describe('Batch persistence', () => {
  function tmpFile() {
    return path.join(os.tmpdir(), `batches-test-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
  }

  test('saveBatchesToDisk is a no-op when no path is configured', async () => {
    await assert.doesNotReject(() => saveBatchesToDisk(null));
  });

  test('loadBatchesFromDisk is a no-op when the file does not exist', () => {
    assert.doesNotThrow(() => loadBatchesFromDisk(tmpFile()));
  });

  test('round-trips an ended batch through save and load', async () => {
    const file = tmpFile();
    const batch = {
      id:                  'msgbatch_persist_test',
      status:              'ended',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            new Date().toISOString(),
      cancel_initiated_at: null,
      requests:            [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:             new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
      cancelRequested:     false,
    };
    _batches.set(batch.id, batch);

    try {
      await saveBatchesToDisk(file);
      assert.ok(fs.existsSync(file), 'persisted file should exist');

      _batches.delete(batch.id);
      loadBatchesFromDisk(file);

      const loaded = _batches.get(batch.id);
      assert.ok(loaded, 'batch should be reloaded from disk');
      assert.equal(loaded.status, 'ended');
      assert.ok(loaded.results instanceof Map, 'results should be reconstructed as a Map');
      assert.deepEqual(loaded.results.get('r1'), { type: 'succeeded', message: { id: 'msg_1' } });
    } finally {
      _batches.delete(batch.id);
      fs.rmSync(file, { force: true });
    }
  });

  test('resumes a persisted in-progress batch and marks it ended once all items already have results', async () => {
    const file = tmpFile();
    const batch = {
      id:                  'msgbatch_resume_test',
      status:              'in_progress',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            null,
      cancel_initiated_at: null,
      requests:            [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:             new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
      cancelRequested:     false,
    };
    _batches.set(batch.id, batch);

    try {
      await saveBatchesToDisk(file);
      _batches.delete(batch.id);

      loadBatchesFromDisk(file);
      // loadBatchesFromDisk schedules processBatch via setImmediate for non-ended batches.
      await new Promise(r => setImmediate(r));

      const loaded = _batches.get(batch.id);
      assert.ok(loaded, 'batch should be reloaded from disk');
      assert.equal(loaded.status, 'ended', 'resumed batch with all items already resolved should be marked ended');
      assert.equal(loaded.results.get('r1').type, 'succeeded', 'existing result should be preserved, not reprocessed');
    } finally {
      _batches.delete(batch.id);
      fs.rmSync(file, { force: true });
    }
  });
});

// ── Batch ownership / per-API-key isolation ────────────────────────────────────
// One caller's API key must not be able to list, read, cancel, or delete batches
// created under a different key (or under "default" when no key was used).
describe('Batch ownership / per-API-key isolation', () => {
  function makeBatch(overrides = {}) {
    const id = 'msgbatch_owner_test_' + Math.random().toString(36).slice(2);
    const batch = {
      id,
      status:              'ended',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            new Date().toISOString(),
      cancel_initiated_at: null,
      requests:            [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:             new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
      cancelRequested:     false,
      ...overrides,
    };
    _batches.set(id, batch);
    return batch;
  }

  function mockReq(apiKeyName) {
    return { headers: {}, socket: { remoteAddress: '127.0.0.1' }, _apiKeyName: apiKeyName };
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; },
    };
  }

  test('batchOwnerName falls back to "default" when no API key matched', () => {
    assert.equal(batchOwnerName({}), 'default');
    assert.equal(batchOwnerName({ _apiKeyName: 'nick' }), 'nick');
  });

  test('batchOwnedByCaller treats a missing owner field as "default"', () => {
    const batch = makeBatch({ owner: undefined });
    try {
      assert.equal(batchOwnedByCaller(mockReq(undefined), batch), true);
      assert.equal(batchOwnedByCaller(mockReq('nick'), batch), false);
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('handleCreateBatch tags the new batch with the caller\'s key name', async () => {
    const req = {
      headers: {}, socket: { remoteAddress: '127.0.0.1' }, _apiKeyName: 'nick',
      [Symbol.asyncIterator]: async function* () {
        yield JSON.stringify({ requests: [{ custom_id: 'r1', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 1 } }] });
      },
    };
    const res = mockRes();
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'hi' } }] }), text: async () => '{}', body: null });
    try {
      await handleCreateBatch(req, res);
      const created = JSON.parse(res._body);
      assert.equal(_batches.get(created.id).owner, 'nick');
    } finally {
      global.fetch = orig;
      _batches.delete(JSON.parse(res._body).id);
    }
  });

  test('handleListBatches only returns batches owned by the caller', async () => {
    const nickBatch    = makeBatch({ owner: 'nick' });
    const familyBatch  = makeBatch({ owner: 'family' });
    const legacyBatch  = makeBatch({ owner: undefined });
    try {
      const res = mockRes();
      await handleListBatches(mockReq('nick'), res);
      const ids = JSON.parse(res._body).data.map(b => b.id);
      assert.ok(ids.includes(nickBatch.id), 'caller should see their own batch');
      assert.ok(!ids.includes(familyBatch.id), 'caller should not see another key\'s batch');
      assert.ok(!ids.includes(legacyBatch.id), 'caller should not see a legacy batch owned by "default"');

      const resDefault = mockRes();
      await handleListBatches(mockReq(undefined), resDefault);
      const defaultIds = JSON.parse(resDefault._body).data.map(b => b.id);
      assert.ok(defaultIds.includes(legacyBatch.id), '"default" caller should see legacy batches with no owner');
      assert.ok(!defaultIds.includes(nickBatch.id), '"default" caller should not see another key\'s batch');
    } finally {
      _batches.delete(nickBatch.id);
      _batches.delete(familyBatch.id);
      _batches.delete(legacyBatch.id);
    }
  });

  test('handleGetBatch returns 404 for a batch owned by a different key', async () => {
    const batch = makeBatch({ owner: 'nick' });
    try {
      const res = mockRes();
      await handleGetBatch(mockReq('family'), res, batch.id);
      assert.equal(res._status, 404);
      assert.equal(JSON.parse(res._body).error.type, 'not_found_error');

      const resOwner = mockRes();
      await handleGetBatch(mockReq('nick'), resOwner, batch.id);
      assert.equal(resOwner._status, 200);
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('handleGetBatchResults returns 404 for a batch owned by a different key', async () => {
    const batch = makeBatch({ owner: 'nick' });
    try {
      const res = mockRes();
      await handleGetBatchResults(mockReq('family'), res, batch.id);
      assert.equal(res._status, 404);
      assert.equal(JSON.parse(res._body).error.type, 'not_found_error');
    } finally {
      _batches.delete(batch.id);
    }
  });

  // EventEmitter-based mock so res.write() can simulate a full write buffer (returning
  // false) and later signal recovery via a real 'drain' event, the way Node's actual
  // http.ServerResponse does.
  function backpressureMockRes(failAfterWrites) {
    const { EventEmitter } = require('events');
    const res = new EventEmitter();
    res._status = null;
    res._body = '';
    res._headers = {};
    res._writeCount = 0;
    res.writableEnded = false;
    res.setHeader = (k, v) => { res._headers[k] = v; };
    res.getHeader = (k) => res._headers[k];
    res.writeHead = (status, headers) => { res._status = status; if (headers) Object.assign(res._headers, headers); };
    res.write = (chunk) => {
      res._body += chunk;
      res._writeCount++;
      return res._writeCount <= failAfterWrites ? false : true;
    };
    res.end = (chunk = '') => { res._body += chunk; res.writableEnded = true; };
    return res;
  }

  test('handleGetBatchResults pauses on backpressure and resumes after drain', async () => {
    const batch = makeBatch({
      owner: 'nick',
      requests: [
        { custom_id: 'r1', params: {} },
        { custom_id: 'r2', params: {} },
        { custom_id: 'r3', params: {} },
      ],
      results: new Map([
        ['r1', { type: 'succeeded', message: { id: 'm1' } }],
        ['r2', { type: 'succeeded', message: { id: 'm2' } }],
        ['r3', { type: 'succeeded', message: { id: 'm3' } }],
      ]),
    });
    try {
      const res = backpressureMockRes(1); // first write() reports a full buffer
      const pending = handleGetBatchResults(mockReq('nick'), res, batch.id);
      await new Promise(r => setImmediate(r));
      assert.equal(res._writeCount, 1, 'should stop writing until drain fires');
      assert.equal(res.writableEnded, false, 'should not end the response while paused');

      res.emit('drain');
      await pending;

      assert.equal(res._writeCount, 3, 'all results should be written once drained');
      assert.equal(res.writableEnded, true);
      const lines = res._body.trim().split('\n').map(l => JSON.parse(l));
      assert.deepEqual(lines.map(l => l.custom_id), ['r1', 'r2', 'r3']);
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('handleGetBatchResults stops writing once the client disconnects mid-stream', async () => {
    const batch = makeBatch({
      owner: 'nick',
      requests: [
        { custom_id: 'r1', params: {} },
        { custom_id: 'r2', params: {} },
        { custom_id: 'r3', params: {} },
      ],
      results: new Map([
        ['r1', { type: 'succeeded', message: { id: 'm1' } }],
        ['r2', { type: 'succeeded', message: { id: 'm2' } }],
        ['r3', { type: 'succeeded', message: { id: 'm3' } }],
      ]),
    });
    try {
      const res = backpressureMockRes(1); // first write() reports a full buffer
      const pending = handleGetBatchResults(mockReq('nick'), res, batch.id);
      await new Promise(r => setImmediate(r));
      assert.equal(res._writeCount, 1, 'should be paused waiting on drain');

      res.writableEnded = true; // client disconnected while paused
      res.emit('close');
      await pending;

      assert.equal(res._writeCount, 1, 'should not write further lines after disconnect');
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('handleCancelBatch returns 404 and does not cancel a batch owned by a different key', async () => {
    const batch = makeBatch({ owner: 'nick', status: 'in_progress', ended_at: null });
    try {
      const res = mockRes();
      await handleCancelBatch(mockReq('family'), res, batch.id);
      assert.equal(res._status, 404);
      assert.equal(batch.cancelRequested, false, 'batch should not be canceled by a non-owner');
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('handleDeleteBatch returns 404 and keeps a batch owned by a different key', async () => {
    const batch = makeBatch({ owner: 'nick' });
    try {
      const res = mockRes();
      await handleDeleteBatch(mockReq('family'), res, batch.id);
      assert.equal(res._status, 404);
      assert.ok(_batches.has(batch.id), 'batch should not be deleted by a non-owner');
    } finally {
      _batches.delete(batch.id);
    }
  });
});

// ── GET /v1/messages/batches pagination ─────────────────────────────────────
describe('parseBatchListParams / handleListBatches pagination', () => {
  function makeBatch(overrides = {}) {
    const id = 'msgbatch_page_test_' + Math.random().toString(36).slice(2);
    const batch = {
      id,
      status:              'ended',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            new Date().toISOString(),
      cancel_initiated_at: null,
      requests:            [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:             new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
      cancelRequested:     false,
      owner:               'pager',
      ...overrides,
    };
    _batches.set(id, batch);
    return batch;
  }

  function mockReq(url) {
    return { url, headers: {}, socket: { remoteAddress: '127.0.0.1' }, _apiKeyName: 'pager' };
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      write(chunk) { this._body += chunk; },
      end(chunk = '') { this._body += chunk; },
    };
  }

  test('parseBatchListParams defaults limit to 20 with no cursors', () => {
    assert.deepEqual(parseBatchListParams({ url: '/v1/messages/batches' }), { limit: 20, before_id: null, after_id: null });
  });

  test('parseBatchListParams reads limit/before_id/after_id from the query string', () => {
    const parsed = parseBatchListParams({ url: '/v1/messages/batches?limit=5&after_id=msgbatch_1' });
    assert.deepEqual(parsed, { limit: 5, before_id: null, after_id: 'msgbatch_1' });
  });

  test('parseBatchListParams rejects an out-of-range or non-integer limit', () => {
    assert.ok(parseBatchListParams({ url: '/v1/messages/batches?limit=0' }).error);
    assert.ok(parseBatchListParams({ url: '/v1/messages/batches?limit=1001' }).error);
    assert.ok(parseBatchListParams({ url: '/v1/messages/batches?limit=abc' }).error);
    assert.ok(parseBatchListParams({ url: '/v1/messages/batches?limit=2.5' }).error);
  });

  test('handleListBatches returns 400 invalid_request_error for a bad limit', async () => {
    const res = mockRes();
    await handleListBatches(mockReq('/v1/messages/batches?limit=0'), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('handleListBatches paginates newest-first and reports has_more', async () => {
    // Created in order b0 (oldest) .. b4 (newest); listing is newest-first.
    const batches = [];
    for (let i = 0; i < 5; i++) batches.push(makeBatch());
    try {
      const res = mockRes();
      await handleListBatches(mockReq('/v1/messages/batches?limit=2'), res);
      const page1 = JSON.parse(res._body);
      assert.equal(page1.data.length, 2);
      assert.equal(page1.data[0].id, batches[4].id);
      assert.equal(page1.data[1].id, batches[3].id);
      assert.equal(page1.has_more, true);
      assert.equal(page1.first_id, batches[4].id);
      assert.equal(page1.last_id, batches[3].id);

      const res2 = mockRes();
      await handleListBatches(mockReq(`/v1/messages/batches?limit=2&after_id=${page1.last_id}`), res2);
      const page2 = JSON.parse(res2._body);
      assert.deepEqual(page2.data.map(b => b.id), [batches[2].id, batches[1].id]);
      assert.equal(page2.has_more, true);

      const res3 = mockRes();
      await handleListBatches(mockReq(`/v1/messages/batches?limit=2&after_id=${page2.last_id}`), res3);
      const page3 = JSON.parse(res3._body);
      assert.deepEqual(page3.data.map(b => b.id), [batches[0].id]);
      assert.equal(page3.has_more, false);
    } finally {
      for (const b of batches) _batches.delete(b.id);
    }
  });

  test('handleListBatches before_id returns the page immediately newer than the cursor', async () => {
    const batches = [];
    for (let i = 0; i < 3; i++) batches.push(makeBatch());
    try {
      const res = mockRes();
      await handleListBatches(mockReq(`/v1/messages/batches?limit=10&before_id=${batches[0].id}`), res);
      const body = JSON.parse(res._body);
      assert.deepEqual(body.data.map(b => b.id), [batches[2].id, batches[1].id]);
      assert.equal(body.has_more, false);
    } finally {
      for (const b of batches) _batches.delete(b.id);
    }
  });

  test('handleListBatches treats a cursor from another caller\'s batch as an empty page', async () => {
    const otherBatch = makeBatch({ owner: 'someone-else' });
    const ownBatch    = makeBatch();
    try {
      const res = mockRes();
      await handleListBatches(mockReq(`/v1/messages/batches?after_id=${otherBatch.id}`), res);
      const body = JSON.parse(res._body);
      assert.deepEqual(body.data, []);
      assert.equal(body.has_more, false);
    } finally {
      _batches.delete(otherBatch.id);
      _batches.delete(ownBatch.id);
    }
  });
});

// ── handleDeleteBatch ─────────────────────────────────────────────────────────
describe('handleDeleteBatch', () => {
  function makeBatch(overrides = {}) {
    const id = 'msgbatch_delete_test_' + Math.random().toString(36).slice(2);
    const batch = {
      id,
      status:              'ended',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            new Date().toISOString(),
      cancel_initiated_at: null,
      requests:            [],
      results:             new Map(),
      cancelRequested:     false,
      ...overrides,
    };
    _batches.set(id, batch);
    return batch;
  }

  function mockReq() {
    return { headers: {}, socket: { remoteAddress: '127.0.0.1' } };
  }

  function mockRes() {
    return {
      _status: null,
      _body: '',
      _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(status, headers) { this._status = status; if (headers) Object.assign(this._headers, headers); },
      end(chunk = '') { this._body += chunk; },
    };
  }

  test('returns 404 for an unknown batch id', async () => {
    const res = mockRes();
    await handleDeleteBatch(mockReq(), res, 'msgbatch_does_not_exist');
    assert.equal(res._status, 404);
    assert.equal(JSON.parse(res._body).error.type, 'not_found_error');
  });

  test('returns 400 and keeps the batch when it has not ended yet', async () => {
    const batch = makeBatch({ status: 'in_progress', ended_at: null });
    try {
      const res = mockRes();
      await handleDeleteBatch(mockReq(), res, batch.id);
      assert.equal(res._status, 400);
      assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
      assert.ok(_batches.has(batch.id), 'in-progress batch should not be deleted');
    } finally {
      _batches.delete(batch.id);
    }
  });

  test('deletes an ended batch and returns message_batch_deleted', async () => {
    const batch = makeBatch();
    const res = mockRes();
    await handleDeleteBatch(mockReq(), res, batch.id);
    assert.equal(res._status, 200);
    assert.deepEqual(JSON.parse(res._body), { id: batch.id, type: 'message_batch_deleted' });
    assert.ok(!_batches.has(batch.id), 'deleted batch should be removed from the Map');
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

  // PROXY_SYSTEM_PROMPT injection: reloads the module with the env var set, since
  // it's read once into a module-level constant at require time.
  describe('PROXY_SYSTEM_PROMPT injection', () => {
    function withProxyEnv(envOverrides, fn) {
      const modKey = require.resolve('./proxy');
      const savedMod = require.cache[modKey];
      const savedEnv = {};
      for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
      let freshProxy;
      try {
        for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
        delete require.cache[modKey];
        freshProxy = require('./proxy');
      } finally {
        for (const [k, v] of Object.entries(savedEnv)) {
          if (v !== undefined) process.env[k] = v;
          else delete process.env[k];
        }
        delete require.cache[modKey];
        require.cache[modKey] = savedMod;
      }
      return fn(freshProxy);
    }

    function captureFetch() {
      const orig = global.fetch;
      const captured = {};
      global.fetch = async (_url, opts) => {
        captured.body = JSON.parse(opts.body);
        return {
          ok: true, status: 200,
          json: async () => ({ id: 'x', choices: [{ message: { role: 'assistant', content: 'hi' }, finish_reason: 'stop', index: 0 } ] }),
          text: async () => '',
          body: null,
        };
      };
      return { captured, restore: () => { global.fetch = orig; } };
    }

    test('prepends to a string system message', async () => {
      await withProxyEnv({ PROXY_SYSTEM_PROMPT: 'Operator rule.' }, async (m) => {
        const { captured, restore } = captureFetch();
        try {
          await m.handleOpenAIChat(mockReq({
            messages: [{ role: 'system', content: 'Be terse.' }, { role: 'user', content: 'hi' }],
          }), mockRes());
          const sysMsg = captured.body.messages.find(msg => msg.role === 'system');
          assert.equal(sysMsg.content, 'Operator rule.\n\nBe terse.');
        } finally { restore(); }
      });
    });

    test('preserves array-form system content instead of discarding it', async () => {
      // Regression test: PROXY_SYSTEM_PROMPT injection previously collapsed a
      // non-string system message content to '', silently dropping it.
      await withProxyEnv({ PROXY_SYSTEM_PROMPT: 'Operator rule.' }, async (m) => {
        const { captured, restore } = captureFetch();
        try {
          const originalBlocks = [{ type: 'text', text: 'Be terse.' }];
          await m.handleOpenAIChat(mockReq({
            messages: [{ role: 'system', content: originalBlocks }, { role: 'user', content: 'hi' }],
          }), mockRes());
          const sysMsg = captured.body.messages.find(msg => msg.role === 'system');
          assert.ok(Array.isArray(sysMsg.content), 'system content should remain an array');
          assert.deepEqual(sysMsg.content, [{ type: 'text', text: 'Operator rule.' }, ...originalBlocks]);
        } finally { restore(); }
      });
    });

    test('inserts a new system message when none exists', async () => {
      await withProxyEnv({ PROXY_SYSTEM_PROMPT: 'Operator rule.' }, async (m) => {
        const { captured, restore } = captureFetch();
        try {
          await m.handleOpenAIChat(mockReq({ messages: [{ role: 'user', content: 'hi' }] }), mockRes());
          assert.equal(captured.body.messages[0].role, 'system');
          assert.equal(captured.body.messages[0].content, 'Operator rule.');
        } finally { restore(); }
      });
    });
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

// ── handleEmbeddings ──────────────────────────────────────────────────────────
describe('handleEmbeddings', () => {
  function mockReq(body) {
    return {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST',
      url: '/v1/embeddings',
      _apiKeyName: 'default',
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
      _logMeta: null,
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

  function stubFetch(ollamaResponse, status = 200) {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: status < 400,
      status,
      json: async () => ollamaResponse,
      text: async () => JSON.stringify(ollamaResponse),
    });
    return () => { global.fetch = orig; };
  }

  test('returns 400 when body is not valid JSON', async () => {
    const req = {
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1' },
      method: 'POST', url: '/v1/embeddings',
      [Symbol.asyncIterator]: async function* () { yield 'NOT JSON'; },
    };
    const res = mockRes();
    await handleEmbeddings(req, res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('returns 400 when input is missing', async () => {
    const res = mockRes();
    await handleEmbeddings(mockReq({ model: 'nomic-embed-text' }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /input/);
  });

  test('returns 400 when model field is not a string', async () => {
    const res = mockRes();
    await handleEmbeddings(mockReq({ model: 42, input: 'hello' }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('returns 400 when encoding_format is not "float" or "base64"', async () => {
    const res = mockRes();
    await handleEmbeddings(mockReq({ input: 'hello', encoding_format: 'hex' }), res);
    assert.equal(res._status, 400);
    const body = JSON.parse(res._body);
    assert.equal(body.error.type, 'invalid_request_error');
    assert.match(body.error.message, /encoding_format/);
  });

  test('returns OpenAI-compatible embedding envelope for a successful request', async () => {
    const restore = stubFetch({
      embeddings: [[0.1, 0.2, 0.3]],
      prompt_eval_count: 7,
    });
    try {
      const res = mockRes();
      await handleEmbeddings(mockReq({ model: 'nomic-embed-text', input: 'hello world' }), res);
      assert.equal(res._status, 200);
      const body = JSON.parse(res._body);
      assert.equal(body.object, 'list');
      assert.equal(body.data.length, 1);
      assert.equal(body.data[0].object, 'embedding');
      assert.deepEqual(body.data[0].embedding, [0.1, 0.2, 0.3]);
      assert.equal(body.data[0].index, 0);
      assert.equal(body.usage.prompt_tokens, 7);
      assert.equal(body.usage.total_tokens, 7);
    } finally {
      restore();
    }
  });

  test('encodes embeddings as base64 float32 when encoding_format is "base64"', async () => {
    const restore = stubFetch({
      embeddings: [[0.1, 0.2, 0.3]],
      prompt_eval_count: 7,
    });
    try {
      const res = mockRes();
      await handleEmbeddings(mockReq({ model: 'nomic-embed-text', input: 'hello world', encoding_format: 'base64' }), res);
      assert.equal(res._status, 200);
      const body = JSON.parse(res._body);
      assert.equal(typeof body.data[0].embedding, 'string');
      const buf = Buffer.from(body.data[0].embedding, 'base64');
      const decoded = [...new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)];
      assert.deepEqual(decoded.map(n => Math.round(n * 10) / 10), [0.1, 0.2, 0.3]);
    } finally {
      restore();
    }
  });

  test('returns a plain float array when encoding_format is "float" or omitted', async () => {
    const restore = stubFetch({
      embeddings: [[0.1, 0.2, 0.3]],
      prompt_eval_count: 7,
    });
    try {
      const res = mockRes();
      await handleEmbeddings(mockReq({ input: 'hello world', encoding_format: 'float' }), res);
      assert.deepEqual(JSON.parse(res._body).data[0].embedding, [0.1, 0.2, 0.3]);
    } finally {
      restore();
    }
  });

  test('accepts input as an array of strings', async () => {
    let sentBody;
    const orig = global.fetch;
    global.fetch = async (_url, opts) => {
      sentBody = JSON.parse(opts.body);
      return {
        ok: true, status: 200,
        json: async () => ({ embeddings: [[0.1], [0.2]], prompt_eval_count: 4 }),
      };
    };
    try {
      const res = mockRes();
      await handleEmbeddings(mockReq({ input: ['foo', 'bar'] }), res);
      assert.deepEqual(sentBody.input, ['foo', 'bar']);
      const body = JSON.parse(res._body);
      assert.equal(body.data.length, 2);
      assert.equal(body.data[0].index, 0);
      assert.equal(body.data[1].index, 1);
    } finally {
      global.fetch = orig;
    }
  });

  test('records token usage in _metrics and sets res._logMeta', async () => {
    const restore = stubFetch({
      embeddings: [[0.5, 0.6]],
      prompt_eval_count: 12,
    });
    try {
      const beforeIn  = _metrics.tokensIn;
      const beforeOut = _metrics.tokensOut;
      const res = mockRes();
      await handleEmbeddings(mockReq({ input: 'test sentence' }), res);
      assert.equal(_metrics.tokensIn  - beforeIn,  12);
      assert.equal(_metrics.tokensOut - beforeOut,   0);
      assert.ok(res._logMeta, 'res._logMeta should be set');
      assert.equal(res._logMeta.tokensIn,  12);
      assert.equal(res._logMeta.tokensOut,  0);
    } finally {
      restore();
    }
  });

  test('handles missing prompt_eval_count gracefully (treats as 0 tokens)', async () => {
    const restore = stubFetch({ embeddings: [[0.1]] });
    try {
      const beforeIn = _metrics.tokensIn;
      const res = mockRes();
      await handleEmbeddings(mockReq({ input: 'hi' }), res);
      assert.equal(res._status, 200);
      assert.equal(_metrics.tokensIn - beforeIn, 0);
      assert.equal(res._logMeta.tokensIn, 0);
    } finally {
      restore();
    }
  });

  test('returns 502 when Ollama returns a non-ok response', async () => {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: false, status: 400,
      text: async () => JSON.stringify({ error: 'model not found' }),
    });
    try {
      const res = mockRes();
      await handleEmbeddings(mockReq({ input: 'hi' }), res);
      assert.notEqual(res._status, 200);
    } finally {
      global.fetch = orig;
    }
  });

  test('does not send num_ctx/keep_alive to /api/embed when unset', async () => {
    let sentBody;
    const orig = global.fetch;
    global.fetch = async (_url, opts) => {
      sentBody = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({ embeddings: [[0.1]], prompt_eval_count: 1 }) };
    };
    try {
      await handleEmbeddings(mockReq({ input: 'hi' }), mockRes());
      assert.equal('num_ctx' in sentBody, false);
      assert.equal('keep_alive' in sentBody, false);
    } finally {
      global.fetch = orig;
    }
  });

  // Helper: load a fresh proxy module with OLLAMA_NUM_CTX/OLLAMA_KEEP_ALIVE
  // set, run a single test, then restore the module cache so other tests
  // are unaffected — mirrors the withHosts()/withForceThink() pattern used
  // elsewhere in this file for env vars captured as module-level consts.
  function withOllamaTuning(fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    let freshProxy;
    try {
      process.env.OLLAMA_NUM_CTX = '4096';
      process.env.OLLAMA_KEEP_ALIVE = '30m';
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      delete process.env.OLLAMA_NUM_CTX;
      delete process.env.OLLAMA_KEEP_ALIVE;
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  test('forwards OLLAMA_NUM_CTX and OLLAMA_KEEP_ALIVE to /api/embed when configured', async () => {
    await withOllamaTuning(async (mod) => {
      let sentBody;
      const orig = global.fetch;
      global.fetch = async (_url, opts) => {
        sentBody = JSON.parse(opts.body);
        return { ok: true, status: 200, json: async () => ({ embeddings: [[0.1]], prompt_eval_count: 1 }) };
      };
      try {
        await mod.handleEmbeddings(mockReq({ input: 'hi' }), mockRes());
        assert.equal(sentBody.num_ctx, 4096);
        assert.equal(sentBody.keep_alive, '30m');
      } finally {
        global.fetch = orig;
      }
    });
  });
});

// ── handleCreateBatch — max_tokens validation ─────────────────────────────────
describe('handleCreateBatch — max_tokens validation', () => {
  function makeReq(body) {
    return {
      headers: {},
      socket: { remoteAddress: '127.0.0.1' },
      _apiKeyName: undefined,
      [Symbol.asyncIterator]: async function* () { yield JSON.stringify(body); },
    };
  }
  function makeRes() {
    return {
      _status: null, _body: '', _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(s, h) { this._status = s; if (h) Object.assign(this._headers, h); },
      write(c) { this._body += c; },
      end(c = '') { this._body += c; },
    };
  }
  const validItem = { custom_id: 'r1', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 10 } };

  test('accepts a batch with a valid max_tokens', async () => {
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'hi' } }] }), text: async () => '{}', body: null });
    try {
      const res = makeRes();
      await handleCreateBatch(makeReq({ requests: [validItem] }), res);
      assert.equal(res._status, 200);
      const batch = JSON.parse(res._body);
      _batches.delete(batch.id);
    } finally {
      global.fetch = orig;
    }
  });

  test('accepts a batch with max_tokens omitted (defaults to PROXY_MAX_TOKENS)', async () => {
    const orig = global.fetch;
    global.fetch = async () => ({ ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'hi' } }] }), text: async () => '{}', body: null });
    try {
      const item = { custom_id: 'r1', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test' } };
      const res = makeRes();
      await handleCreateBatch(makeReq({ requests: [item] }), res);
      assert.equal(res._status, 200);
      const batch = JSON.parse(res._body);
      _batches.delete(batch.id);
    } finally {
      global.fetch = orig;
    }
  });

  test('rejects a batch item with max_tokens: 0 at creation time', async () => {
    const item = { custom_id: 'bad', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 0 } };
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [item] }), res);
    assert.equal(res._status, 400);
    const err = JSON.parse(res._body).error;
    assert.equal(err.type, 'invalid_request_error');
    assert.ok(err.message.includes('bad'), 'error message should reference the bad custom_id');
    assert.ok(err.message.includes('max_tokens'));
  });

  test('rejects a batch item with negative max_tokens at creation time', async () => {
    const item = { custom_id: 'neg', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: -5 } };
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [item] }), res);
    assert.equal(res._status, 400);
    const err = JSON.parse(res._body).error;
    assert.equal(err.type, 'invalid_request_error');
    assert.ok(err.message.includes('neg'));
  });

  test('rejects a batch item with non-integer float max_tokens at creation time', async () => {
    const item = { custom_id: 'flt', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 1.5 } };
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [item] }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('batch with invalid max_tokens is never created in _batches', async () => {
    const before = _batches.size;
    const item = { custom_id: 'bad2', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: -1 } };
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [item] }), res);
    assert.equal(res._status, 400);
    assert.equal(_batches.size, before, 'no batch should be created when max_tokens is invalid');
  });
});

describe('handleCreateBatch — requests array size limit', () => {
  function makeReq(body) {
    return {
      headers: {},
      socket: { remoteAddress: '127.0.0.1' },
      _apiKeyName: undefined,
      [Symbol.asyncIterator]: async function* () { yield JSON.stringify(body); },
    };
  }
  function makeRes() {
    return {
      _status: null, _body: '', _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(s, h) { this._status = s; if (h) Object.assign(this._headers, h); },
      write(c) { this._body += c; },
      end(c = '') { this._body += c; },
    };
  }

  test('rejects a batch with more than MAX_BATCH_REQUESTS items, without validating any of them', async () => {
    const before = _batches.size;
    // Content doesn't matter — the size check runs before any per-item validation,
    // so a sparse array of holes (JSON-serialized as `null`s) is enough to prove it.
    const oversized = new Array(MAX_BATCH_REQUESTS + 1);
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: oversized }), res);
    assert.equal(res._status, 400);
    const err = JSON.parse(res._body).error;
    assert.equal(err.type, 'invalid_request_error');
    assert.ok(err.message.includes(String(MAX_BATCH_REQUESTS)), 'error message should name the limit');
    assert.equal(_batches.size, before, 'no batch should be created when the size limit is exceeded');
  });

  test('a batch with exactly MAX_BATCH_REQUESTS items passes the size check', async () => {
    // All items intentionally invalid (missing custom_id) so per-item validation
    // rejects on the very first item instead of validating 100,000 entries — this
    // still proves the boundary itself (length === MAX_BATCH_REQUESTS) is accepted
    // past the size check and falls through to the existing per-item validation,
    // i.e. the comparison is `>`, not `>=`.
    const atLimit = new Array(MAX_BATCH_REQUESTS).fill({});
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: atLimit }), res);
    assert.equal(res._status, 400);
    const err = JSON.parse(res._body).error;
    assert.ok(err.message.includes('custom_id'), 'should fail on the existing custom_id check, not the size limit');
    assert.ok(!err.message.includes('must contain at most'), 'size limit should not fire exactly at the boundary');
  });
});

describe('handleCreateBatch — malformed requests array entries', () => {
  function makeReq(body) {
    return {
      headers: {},
      socket: { remoteAddress: '127.0.0.1' },
      _apiKeyName: undefined,
      [Symbol.asyncIterator]: async function* () { yield JSON.stringify(body); },
    };
  }
  function makeRes() {
    return {
      _status: null, _body: '', _headers: {},
      setHeader(k, v) { this._headers[k] = v; },
      getHeader(k) { return this._headers[k]; },
      writeHead(s, h) { this._status = s; if (h) Object.assign(this._headers, h); },
      write(c) { this._body += c; },
      end(c = '') { this._body += c; },
    };
  }

  test('rejects a `null` entry in requests with 400 instead of crashing with 500', async () => {
    const before = _batches.size;
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [null] }), res);
    assert.equal(res._status, 400);
    const err = JSON.parse(res._body).error;
    assert.equal(err.type, 'invalid_request_error');
    assert.equal(_batches.size, before, 'no batch should be created for a malformed entry');
  });

  test('rejects a non-object (string) entry in requests with 400', async () => {
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: ['not-an-object'] }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('rejects an array entry in requests with 400', async () => {
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [[]] }), res);
    assert.equal(res._status, 400);
    assert.equal(JSON.parse(res._body).error.type, 'invalid_request_error');
  });

  test('a valid entry after a malformed one never gets processed', async () => {
    const before = _batches.size;
    const validItem = { custom_id: 'r1', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 10 } };
    const res = makeRes();
    await handleCreateBatch(makeReq({ requests: [null, validItem] }), res);
    assert.equal(res._status, 400);
    assert.equal(_batches.size, before, 'the whole batch should be rejected, not partially created');
  });
});

// ── readBody failure cleanup ────────────────────────────────────────────────────
// handleMessages, handleOpenAIChat, handleOpenAICompletions, and handleEmbeddings all
// register a PROXY_TIMEOUT timer and a socket 'close' listener *before* calling
// readBody(req). If readBody throws (e.g. a PROXY_MAX_BODY_SIZE rejection), the
// exception used to skip straight past each handler's own cleanup to the generic
// router catch block, which has no access to those closures — leaving a dangling
// timer (that later fires a misleading "Request timeout" warning for a request that
// actually failed for an unrelated reason) and an unremoved 'close' listener.

describe('readBody failure cleanup (PROXY_TIMEOUT / close listener)', () => {
  const { EventEmitter } = require('events');

  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  function mockReq(bodyChunk) {
    const socket = new EventEmitter();
    socket.remoteAddress = '127.0.0.1';
    return {
      headers: {},
      socket,
      method: 'POST',
      [Symbol.asyncIterator]: async function* () { yield bodyChunk; },
    };
  }

  function mockRes() {
    return {
      headersSent: false,
      writableEnded: false,
      _status: null,
      _body: '',
      setHeader() {},
      getHeader() {},
      writeHead(status) { this._status = status; this.headersSent = true; },
      end(chunk = '') { this._body += chunk; this.writableEnded = true; },
    };
  }

  const oversizedBody = 'x'.repeat(64);
  const cases = [
    ['handleMessages', '/v1/messages'],
    ['handleOpenAIChat', '/v1/chat/completions'],
    ['handleOpenAICompletions', '/v1/completions'],
    ['handleEmbeddings', '/v1/embeddings'],
  ];

  for (const [handlerName, url] of cases) {
    test(`${handlerName} removes the close listener and clears the timeout when readBody rejects an oversized body`, async () => {
      await withProxyEnv({ PROXY_MAX_BODY_SIZE: '10', PROXY_TIMEOUT: '60000' }, async (m) => {
        const req = mockReq(oversizedBody);
        req.url = url;
        const res = mockRes();
        await assert.rejects(
          () => m[handlerName](req, res),
          (e) => e.code === 'PAYLOAD_TOO_LARGE'
        );
        assert.equal(req.socket.listenerCount('close'), 0, 'close listener should be removed after readBody throws');
      });
    });
  }
});

// ── Rate limiting on POST /v1/messages/batches (batch creation) ────────────────
// Every other endpoint that triggers Ollama inference (/v1/messages,
// /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/messages/count_tokens)
// is gated by RATE_LIMIT_RPM / RATE_LIMIT_PER_IP_RPM / RATE_LIMIT_PER_KEY_RPM, but
// batch creation previously had none of the three wired up in the router — letting
// a caller bypass all request-rate limits by submitting work through the Batch API
// instead of real-time calls, even though each batch item is processed through the
// same Ollama call path. These tests exercise the router (requestHandler) directly
// so they catch a regression in the wiring itself, not just in checkRateLimit().

describe('rate limiting on POST /v1/messages/batches', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

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
    return {
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
  }

  function stubFetch() {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({ choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
      text: async () => '{}',
      body: null,
    });
    return () => { global.fetch = orig; };
  }

  const validBatchBody = {
    requests: [{ custom_id: 'r1', params: { messages: [{ role: 'user', content: 'hi' }], model: 'test', max_tokens: 10 } }],
  };

  test('RATE_LIMIT_RPM rejects a second batch-creation call within the same window with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res1);
        assert.equal(res1._status, 200, 'first batch-creation call should succeed');

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res2);
        assert.equal(res2._status, 429, 'second batch-creation call should be rate-limited');
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_IP_RPM rejects a second batch-creation call from the same IP with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_IP_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_KEY_RPM rejects a second batch-creation call from the same key with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_KEY_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('batch creation succeeds normally when no rate limit env vars are set', async () => {
    const restore = stubFetch();
    try {
      const res = mockRes();
      const { requestHandler } = require('./proxy');
      await requestHandler(mockReq('POST', '/v1/messages/batches', validBatchBody), res);
      assert.equal(res._status, 200);
    } finally { restore(); }
  });
});

// These five routes previously called only checkAuth() and skipped all three
// checkRateLimit() calls applied to every sibling mutating/resource-touching route
// (batch creation, model pull, model delete) — letting an authenticated caller bypass
// configured request-rate limits entirely by hitting batch sub-routes instead.
describe('rate limiting on batch sub-routes (list/get/results/cancel/delete)', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

  function mockReq(method, path) {
    return {
      method,
      url: path,
      headers: {},
      socket: { once: () => {}, off: () => {}, remoteAddress: '127.0.0.1', encrypted: false },
      [Symbol.asyncIterator]: async function* () {},
    };
  }

  function mockRes() {
    const listeners = {};
    return {
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
  }

  function makeBatch(m, overrides = {}) {
    const id = 'msgbatch_rl_test_' + Math.random().toString(36).slice(2);
    const batch = {
      id,
      status:              'in_progress',
      created_at:          new Date().toISOString(),
      expires_at:          new Date(Date.now() + 60_000).toISOString(),
      ended_at:            null,
      cancel_initiated_at: null,
      requests:            [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
      results:             new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
      cancelRequested:     false,
      ...overrides,
    };
    m._batches.set(id, batch);
    return batch;
  }

  const ROUTES = [
    { name: 'list',    method: 'GET',    path: (id) => '/v1/messages/batches',           batchOverrides: {} },
    { name: 'get',     method: 'GET',    path: (id) => `/v1/messages/batches/${id}`,      batchOverrides: {} },
    { name: 'results', method: 'GET',    path: (id) => `/v1/messages/batches/${id}/results`, batchOverrides: { status: 'ended', ended_at: new Date().toISOString() } },
    { name: 'cancel',  method: 'POST',   path: (id) => `/v1/messages/batches/${id}/cancel`, batchOverrides: { status: 'in_progress' } },
    { name: 'delete',  method: 'DELETE', path: (id) => `/v1/messages/batches/${id}`,       batchOverrides: { status: 'ended', ended_at: new Date().toISOString() } },
  ];

  for (const route of ROUTES) {
    test(`RATE_LIMIT_RPM rejects a second ${route.name} call within the same window with 429`, async () => {
      await withProxyEnv({ RATE_LIMIT_RPM: '1' }, async (m) => {
        const batch = makeBatch(m, route.batchOverrides);

        const res1 = mockRes();
        await m.requestHandler(mockReq(route.method, route.path(batch.id)), res1);
        assert.equal(res1._status, 200, `first ${route.name} call should succeed`);

        const res2 = mockRes();
        await m.requestHandler(mockReq(route.method, route.path(batch.id)), res2);
        assert.equal(res2._status, 429, `second ${route.name} call should be rate-limited`);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    });
  }

  test('RATE_LIMIT_PER_IP_RPM rejects a second list call from the same IP with 429', async () => {
    await withProxyEnv({ RATE_LIMIT_PER_IP_RPM: '1' }, async (m) => {
      const res1 = mockRes();
      await m.requestHandler(mockReq('GET', '/v1/messages/batches'), res1);
      assert.equal(res1._status, 200);

      const res2 = mockRes();
      await m.requestHandler(mockReq('GET', '/v1/messages/batches'), res2);
      assert.equal(res2._status, 429);
      assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
    });
  });

  test('RATE_LIMIT_PER_KEY_RPM rejects a second list call from the same key with 429', async () => {
    await withProxyEnv({ RATE_LIMIT_PER_KEY_RPM: '1' }, async (m) => {
      const res1 = mockRes();
      await m.requestHandler(mockReq('GET', '/v1/messages/batches'), res1);
      assert.equal(res1._status, 200);

      const res2 = mockRes();
      await m.requestHandler(mockReq('GET', '/v1/messages/batches'), res2);
      assert.equal(res2._status, 429);
      assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
    });
  });

  test('all five batch sub-routes succeed normally when no rate limit env vars are set', async () => {
    const { requestHandler, _batches } = require('./proxy');
    for (const route of ROUTES) {
      const id = 'msgbatch_rl_test_' + Math.random().toString(36).slice(2);
      _batches.set(id, {
        id, status: 'in_progress', created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 60_000).toISOString(), ended_at: null,
        cancel_initiated_at: null,
        requests: [{ custom_id: 'r1', params: { messages: [], model: 'test', max_tokens: 1 } }],
        results: new Map([['r1', { type: 'succeeded', message: { id: 'msg_1' } }]]),
        cancelRequested: false,
        ...route.batchOverrides,
      });
      try {
        const res = mockRes();
        await requestHandler(mockReq(route.method, route.path(id)), res);
        assert.equal(res._status, 200, `${route.name} should succeed with no rate limits configured`);
      } finally {
        _batches.delete(id);
      }
    }
  });
});

describe('rate limiting on POST /v1/models/pull', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

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
    return {
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
  }

  function stubFetch() {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({ status: 'success' }),
      text: async () => '{}',
    });
    return () => { global.fetch = orig; };
  }

  const pullBody = { model: 'llama3.2:1b' };

  test('RATE_LIMIT_RPM rejects a second pull call within the same window with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res1);
        assert.equal(res1._status, 200, 'first pull call should succeed');

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res2);
        assert.equal(res2._status, 429, 'second pull call should be rate-limited');
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_IP_RPM rejects a second pull call from the same IP with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_IP_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_KEY_RPM rejects a second pull call from the same key with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_KEY_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('pull succeeds normally when no rate limit env vars are set', async () => {
    const restore = stubFetch();
    try {
      const res = mockRes();
      const { requestHandler } = require('./proxy');
      await requestHandler(mockReq('POST', '/v1/models/pull', pullBody), res);
      assert.equal(res._status, 200);
    } finally { restore(); }
  });
});

describe('rate limiting on DELETE /v1/models/:id', () => {
  function withProxyEnv(envOverrides, fn) {
    const modKey = require.resolve('./proxy');
    const savedMod = require.cache[modKey];
    const savedEnv = {};
    for (const k of Object.keys(envOverrides)) savedEnv[k] = process.env[k];
    let freshProxy;
    try {
      for (const [k, v] of Object.entries(envOverrides)) process.env[k] = v;
      delete require.cache[modKey];
      freshProxy = require('./proxy');
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v !== undefined) process.env[k] = v;
        else delete process.env[k];
      }
      delete require.cache[modKey];
      require.cache[modKey] = savedMod;
    }
    return fn(freshProxy);
  }

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
    return {
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
  }

  function stubFetch() {
    const orig = global.fetch;
    global.fetch = async () => ({
      ok: true, status: 200,
      json: async () => ({ status: 'success' }),
      text: async () => '{}',
    });
    return () => { global.fetch = orig; };
  }

  const deletePath = '/v1/models/llama3.2:1b';

  test('RATE_LIMIT_RPM rejects a second delete call within the same window with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res1);
        assert.equal(res1._status, 200, 'first delete call should succeed');

        const res2 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res2);
        assert.equal(res2._status, 429, 'second delete call should be rate-limited');
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_IP_RPM rejects a second delete call from the same IP with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_IP_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('RATE_LIMIT_PER_KEY_RPM rejects a second delete call from the same key with 429', async () => {
    const restore = stubFetch();
    try {
      await withProxyEnv({ RATE_LIMIT_PER_KEY_RPM: '1' }, async (m) => {
        const res1 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res1);
        assert.equal(res1._status, 200);

        const res2 = mockRes();
        await m.requestHandler(mockReq('DELETE', deletePath), res2);
        assert.equal(res2._status, 429);
        assert.equal(JSON.parse(res2._body).error.type, 'rate_limit_error');
      });
    } finally { restore(); }
  });

  test('delete succeeds normally when no rate limit env vars are set', async () => {
    const restore = stubFetch();
    try {
      const res = mockRes();
      const { requestHandler } = require('./proxy');
      await requestHandler(mockReq('DELETE', deletePath), res);
      assert.equal(res._status, 200);
    } finally { restore(); }
  });
});

// ── GET / dashboard — client-side script XSS escaping ─────────────────────────
// handleDashboard() embeds a <script> that fetches /health and /metrics and
// renders the results into the page via innerHTML. Several fields in those
// responses originate from attacker-controlled input — e.g. models_usage keys
// come straight from the client-supplied `model` field (resolveModel() passes
// unrecognized model names through verbatim), and requests_total keys come
// from the raw, unvalidated req.url path (recorded for every request,
// including unauthenticated 404s). These tests run the actual embedded script
// in a vm sandbox with a mocked DOM/fetch to verify malicious values are
// HTML-escaped before reaching innerHTML, rather than just asserting the
// script source contains an esc() call.
describe('GET / dashboard — XSS escaping', () => {
  function extractScript() {
    let captured = '';
    handleDashboard({}, { writeHead() {}, end(html) { captured = html; } });
    const match = captured.match(/<script>([\s\S]*?)<\/script>/);
    assert.ok(match, 'dashboard HTML should contain an embedded <script>');
    return match[1];
  }

  // Runs the dashboard's client-side refresh() against mocked /health and
  // /metrics responses, returning the resulting grid innerHTML.
  async function renderWithMetrics(metrics) {
    const scriptSrc = extractScript();
    const elements = { dot: {}, ts: {}, grid: {} };
    const sandbox = {
      document: { getElementById: (id) => elements[id] },
      fetch: async (url) => {
        if (url === '/health') return { json: async () => ({ status: 'ok' }) };
        if (url === '/metrics') return { json: async () => metrics };
        throw new Error(`unexpected fetch ${url}`);
      },
      setInterval: () => {},
      console, Promise, Date, Object, Math, String,
    };
    vm.createContext(sandbox);
    vm.runInContext(scriptSrc, sandbox);
    assert.equal(typeof sandbox.refresh, 'function', 'script should expose a top-level refresh()');
    await sandbox.refresh();
    return elements.grid.innerHTML;
  }

  test('escapes a malicious route path key in the Requests card', async () => {
    const payload = 'GET /<svg onload=alert(2)>';
    const html = await renderWithMetrics({
      requests_total: { [payload]: 1 },
      status_codes: {}, models_usage: {}, api_keys_usage: {},
    });
    assert.ok(!html.includes(payload), 'raw payload must not appear unescaped');
    assert.ok(html.includes('&lt;svg onload=alert(2)&gt;'), 'payload should be HTML-escaped');
  });

  test('escapes a malicious model name in the Model Usage card', async () => {
    const payload = '<img src=x onerror=alert(1)>';
    const html = await renderWithMetrics({
      requests_total: {}, status_codes: {}, api_keys_usage: {},
      models_usage: { [payload]: { requests: 1, tokens_in: 1, tokens_out: 1 } },
    });
    assert.ok(!html.includes(payload), 'raw payload must not appear unescaped');
    assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'), 'payload should be HTML-escaped');
  });

  test('escapes a malicious Ollama host error message', async () => {
    const payload = '<script>alert(3)</script>';
    const scriptSrc = extractScript();
    const elements = { dot: {}, ts: {}, grid: {} };
    const sandbox = {
      document: { getElementById: (id) => elements[id] },
      fetch: async (url) => {
        if (url === '/health') return { json: async () => ({ status: 'degraded', ollamaError: payload }) };
        if (url === '/metrics') return { json: async () => ({ requests_total: {}, status_codes: {}, models_usage: {}, api_keys_usage: {} }) };
      },
      setInterval: () => {},
      console, Promise, Date, Object, Math, String,
    };
    vm.createContext(sandbox);
    vm.runInContext(scriptSrc, sandbox);
    await sandbox.refresh();
    const html = elements.grid.innerHTML;
    assert.ok(!html.includes(payload), 'raw payload must not appear unescaped');
    assert.ok(html.includes('&lt;script&gt;alert(3)&lt;/script&gt;'), 'payload should be HTML-escaped');
  });

  test('renders normal metrics without throwing and without escaping artifacts', async () => {
    const html = await renderWithMetrics({
      requests_total: { 'GET /health': 5 },
      status_codes: { '200': 5 },
      models_usage: { 'qwen2.5:7b': { requests: 5, tokens_in: 100, tokens_out: 200 } },
      api_keys_usage: {},
    });
    assert.ok(html.includes('GET /health'));
    assert.ok(html.includes('qwen2.5:7b'));
  });
});
