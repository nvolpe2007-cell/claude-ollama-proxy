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
