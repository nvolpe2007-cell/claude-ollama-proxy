'use strict';

const crypto = require('crypto');
const https = require('https');

const ADMIN_TOKEN = process.env.NEWSLETTER_ADMIN_TOKEN || '';
const RESEND_API_KEY = process.env.RESEND_API_KEY || '';
const FROM_EMAIL = process.env.NEWSLETTER_FROM_EMAIL || 'newsletter@localhost';
const SITE_URL = (process.env.PUBLIC_SITE_URL || 'http://localhost:4000').replace(/\/$/, '');

// In-memory store: email → { email, token, subscribedAt, unsubscribedAt|null }
const subscribers = new Map();

// ── Helpers ──────────────────────────────────────────────────────────────────

function genToken() {
  return crypto.randomBytes(32).toString('hex');
}

function json(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

function html(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(body);
}

function checkAdmin(req, res) {
  if (!ADMIN_TOKEN) {
    json(res, 500, { error: 'NEWSLETTER_ADMIN_TOKEN is not configured' });
    return false;
  }
  const auth = (req.headers.authorization || '').trim();
  if (auth !== `Bearer ${ADMIN_TOKEN}`) {
    json(res, 401, { error: 'unauthorized' });
    return false;
  }
  return true;
}

async function sendViaResend(to, subject, htmlBody) {
  if (!RESEND_API_KEY) throw new Error('RESEND_API_KEY not configured');

  const payload = JSON.stringify({ from: FROM_EMAIL, to: [to], subject, html: htmlBody });
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: 'api.resend.com',
        path: '/emails',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${RESEND_API_KEY}`,
          'Content-Length': Buffer.byteLength(payload),
        },
      },
      (res) => {
        let buf = '';
        res.on('data', (c) => (buf += c));
        res.on('end', () => {
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve(JSON.parse(buf));
          } else {
            reject(new Error(`Resend ${res.statusCode}: ${buf}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.end(payload);
  });
}

// ── Route handlers ────────────────────────────────────────────────────────────

async function handleSubscribe(req, res, rawBody) {
  let data;
  try {
    data = JSON.parse(rawBody);
  } catch {
    json(res, 400, { error: 'invalid JSON' });
    return;
  }

  const email = (data.email || '').trim().toLowerCase();
  if (!email || !email.includes('@')) {
    json(res, 400, { error: 'valid email required' });
    return;
  }

  const existing = subscribers.get(email);
  if (existing && !existing.unsubscribedAt) {
    json(res, 200, { ok: true, message: 'already subscribed' });
    return;
  }

  // Re-subscribe or new subscriber
  const token = genToken();
  subscribers.set(email, {
    email,
    token,
    subscribedAt: new Date().toISOString(),
    unsubscribedAt: null,
  });

  if (RESEND_API_KEY) {
    sendViaResend(email, 'Welcome to The Worthy Horse News!', welcomeHtml(token)).catch((e) =>
      console.error('[newsletter] welcome email failed:', e.message)
    );
  }

  json(res, 201, { ok: true });
}

async function handleUnsubscribe(req, res, parsedUrl) {
  const token = parsedUrl.searchParams.get('token') || '';
  const sub = token && [...subscribers.values()].find((s) => s.token === token);

  if (!sub) {
    html(res, 404, pageHtml('Unsubscribe', '<p>Token not found or already expired.</p>'));
    return;
  }

  if (!sub.unsubscribedAt) {
    sub.unsubscribedAt = new Date().toISOString();
  }

  html(
    res,
    200,
    pageHtml(
      'Unsubscribed',
      `<p>You've been unsubscribed from <strong>The Worthy Horse News</strong>.</p>
       <p>We're sorry to see you go. You won't receive any more emails from us.</p>`
    )
  );
}

async function handleAdminStats(req, res) {
  if (!checkAdmin(req, res)) return;

  const all = [...subscribers.values()];
  const active = all.filter((s) => !s.unsubscribedAt);

  json(res, 200, {
    total: all.length,
    active: active.length,
    unsubscribed: all.length - active.length,
  });
}

async function handleAdminDispatch(req, res, rawBody) {
  if (!checkAdmin(req, res)) return;

  let data;
  try {
    data = JSON.parse(rawBody);
  } catch {
    json(res, 400, { error: 'invalid JSON' });
    return;
  }

  const { subject, body: emailBody, preheader, testEmail } = data;
  if (!subject || !emailBody) {
    json(res, 400, { error: 'subject and body are required' });
    return;
  }

  if (!RESEND_API_KEY) {
    json(res, 503, { error: 'RESEND_API_KEY not configured — cannot send emails' });
    return;
  }

  if (testEmail) {
    try {
      await sendViaResend(testEmail, subject, dispatchHtml(emailBody, preheader, testEmail, ''));
      json(res, 200, { ok: true, sent: 1, testEmail });
    } catch (e) {
      json(res, 502, { error: e.message });
    }
    return;
  }

  const active = [...subscribers.values()].filter((s) => !s.unsubscribedAt);
  let sent = 0;
  let errors = 0;

  for (const sub of active) {
    try {
      await sendViaResend(
        sub.email,
        subject,
        dispatchHtml(emailBody, preheader, sub.email, sub.token)
      );
      sent++;
    } catch (e) {
      console.error(`[newsletter] dispatch failed for ${sub.email}:`, e.message);
      errors++;
    }
  }

  json(res, 200, { ok: true, sent, errors, total: active.length });
}

async function handleAdminPage(req, res, parsedUrl) {
  // Password delivered as Bearer token in Authorization header OR as `pw` query param
  // (query-param path is used by the login form and the JS fetch calls from the page).
  const qpw = parsedUrl.searchParams.get('pw') || '';
  const bearer = (req.headers.authorization || '').replace(/^Bearer\s+/i, '');
  const provided = bearer || qpw;

  if (!ADMIN_TOKEN || provided !== ADMIN_TOKEN) {
    html(res, 200, loginHtml(!!provided));
    return;
  }

  html(res, 200, adminHtml(provided));
}

// ── HTML templates ────────────────────────────────────────────────────────────

function pageHtml(title, body) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title} — The Worthy Horse News</title>
  <style>
    body { font-family: Georgia, serif; max-width: 600px; margin: 80px auto; padding: 0 20px; color: #222; }
    h1 { color: #8b6914; }
  </style>
</head>
<body>
  <h1>${title}</h1>
  ${body}
</body>
</html>`;
}

function welcomeHtml(token) {
  const unsub = `${SITE_URL}/api/newsletter/unsubscribe?token=${token}`;
  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8" /><title>Welcome to The Worthy Horse News</title></head>
<body style="font-family:Georgia,serif;color:#222;max-width:600px;margin:0 auto;padding:20px">
  <h1 style="color:#8b6914">Welcome to The Worthy Horse News!</h1>
  <p>Thank you for subscribing. You'll receive updates on equine wellness, bodywork, and holistic horsemanship.</p>
  <hr style="border:1px solid #d4a827;margin:30px 0" />
  <p style="font-size:12px;color:#888">
    You're receiving this because you subscribed at ${SITE_URL}.<br />
    <a href="${unsub}" style="color:#8b6914">Unsubscribe</a>
  </p>
</body>
</html>`;
}

function dispatchHtml(body, preheader, email, token) {
  const preheaderSnippet = preheader
    ? `<div style="display:none;max-height:0;overflow:hidden;mso-hide:all">${preheader}</div>`
    : '';
  const unsubLine = token
    ? `<a href="${SITE_URL}/api/newsletter/unsubscribe?token=${token}" style="color:#8b6914">Unsubscribe</a>`
    : `<span style="color:#aaa">Sent to ${email}</span>`;
  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8" /><title>The Worthy Horse News</title></head>
<body style="font-family:Georgia,serif;color:#222;max-width:600px;margin:0 auto;padding:20px">
  ${preheaderSnippet}
  <div>${body}</div>
  <hr style="border:1px solid #d4a827;margin:30px 0" />
  <p style="font-size:12px;color:#888">${unsubLine}</p>
</body>
</html>`;
}

function loginHtml(failed) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Newsletter Admin — Login</title>
  <style>
    body { font-family: Georgia, serif; max-width: 400px; margin: 120px auto; padding: 0 20px; color: #222; }
    h1 { color: #8b6914; margin-bottom: 24px; }
    label { display: block; margin-bottom: 8px; font-size: 14px; }
    input[type=password] { width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
    button { margin-top: 12px; width: 100%; padding: 10px; background: #c9a227; color: #fff; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
    button:hover { background: #a88520; }
    .err { color: #c00; font-size: 14px; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Newsletter Admin</h1>
  <form method="get" action="/admin/newsletter">
    <label for="pw">Admin password</label>
    <input type="password" id="pw" name="pw" autofocus required />
    ${failed ? '<p class="err">Incorrect password.</p>' : ''}
    <button type="submit">Sign in</button>
  </form>
</body>
</html>`;
}

function adminHtml(pw) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Newsletter Admin — The Worthy Horse News</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: Georgia, serif; background: #0f0f0f; color: #f0e8d0; margin: 0; padding: 40px 20px; min-height: 100vh; }
    h1 { color: #d4a827; font-size: 1.8rem; margin-bottom: 8px; }
    .sub { color: #888; font-size: 14px; margin-bottom: 40px; }
    .card { background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 28px; max-width: 780px; margin: 0 auto 28px; }
    h2 { color: #d4a827; font-size: 1.1rem; margin: 0 0 18px; }
    .stats { display: flex; gap: 24px; flex-wrap: wrap; }
    .stat { background: #111; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px 24px; text-align: center; min-width: 120px; }
    .stat-num { font-size: 2rem; color: #d4a827; font-weight: bold; }
    .stat-lbl { font-size: 12px; color: #888; margin-top: 4px; }
    label { display: block; font-size: 13px; color: #aaa; margin-bottom: 6px; margin-top: 16px; }
    input[type=text], input[type=email], textarea {
      width: 100%; padding: 10px 12px; background: #111; border: 1px solid #333;
      border-radius: 6px; color: #f0e8d0; font-size: 15px; font-family: inherit;
    }
    input[type=text]:focus, input[type=email]:focus, textarea:focus {
      outline: none; border-color: #d4a827;
    }
    textarea { resize: vertical; min-height: 160px; }
    .actions { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
    .btn { padding: 10px 22px; border: none; border-radius: 6px; font-size: 15px; cursor: pointer; font-family: inherit; }
    .btn-primary { background: #c9a227; color: #0f0f0f; }
    .btn-primary:hover { background: #d4a827; }
    .btn-secondary { background: #2a2a2a; color: #f0e8d0; border: 1px solid #444; }
    .btn-secondary:hover { background: #333; }
    .result { margin-top: 14px; font-size: 13px; color: #aaa; white-space: pre-wrap; font-family: monospace; background: #111; border: 1px solid #2a2a2a; border-radius: 6px; padding: 12px; display: none; }
    .result.visible { display: block; }
    .result.ok { border-color: #2a6b2a; color: #7ec87e; }
    .result.err { border-color: #6b2a2a; color: #e07e7e; }
  </style>
</head>
<body>
<div style="max-width:780px;margin:0 auto">
  <h1>The Worthy Horse News</h1>
  <p class="sub">Newsletter admin</p>

  <div class="card">
    <h2>Subscriber stats</h2>
    <div class="stats" id="stats">
      <div class="stat"><div class="stat-num" id="s-active">—</div><div class="stat-lbl">Active</div></div>
      <div class="stat"><div class="stat-num" id="s-total">—</div><div class="stat-lbl">Total</div></div>
      <div class="stat"><div class="stat-num" id="s-unsub">—</div><div class="stat-lbl">Unsubscribed</div></div>
    </div>
    <div class="actions">
      <button class="btn btn-secondary" onclick="loadStats()">Refresh</button>
    </div>
  </div>

  <div class="card">
    <h2>Send newsletter</h2>
    <form id="dispatch-form" onsubmit="dispatch(event)">
      <label for="subject">Subject *</label>
      <input type="text" id="subject" placeholder="e.g. July update from Susie" required />

      <label for="preheader">Preheader (inbox preview text)</label>
      <input type="text" id="preheader" placeholder="Optional — shown in inbox alongside subject" />

      <label for="body">Email body (HTML) *</label>
      <textarea id="body" placeholder="<p>Hi, ...</p>" required></textarea>

      <label for="testEmail">Test email address</label>
      <input type="email" id="testEmail" placeholder="Leave blank to broadcast to all active subscribers" />

      <div class="actions">
        <button type="submit" class="btn btn-primary" id="send-btn">Send</button>
      </div>
    </form>
    <div class="result" id="dispatch-result"></div>
  </div>
</div>

<script>
const PW = ${JSON.stringify(pw)};
const authHeaders = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + PW
};

async function loadStats() {
  try {
    const r = await fetch('/api/newsletter/admin/stats', { headers: authHeaders });
    const d = await r.json();
    document.getElementById('s-active').textContent = d.active ?? '—';
    document.getElementById('s-total').textContent = d.total ?? '—';
    document.getElementById('s-unsub').textContent = d.unsubscribed ?? '—';
  } catch (e) {
    console.error('stats load failed', e);
  }
}

async function dispatch(e) {
  e.preventDefault();
  const btn = document.getElementById('send-btn');
  const resultEl = document.getElementById('dispatch-result');
  const testEmail = document.getElementById('testEmail').value.trim();

  const confirmed = testEmail
    ? true
    : confirm('Send to ALL active subscribers?');
  if (!confirmed) return;

  btn.disabled = true;
  btn.textContent = 'Sending…';
  resultEl.className = 'result';
  resultEl.style.display = 'none';

  try {
    const body = {
      subject: document.getElementById('subject').value,
      body: document.getElementById('body').value,
      preheader: document.getElementById('preheader').value || undefined,
      testEmail: testEmail || undefined,
    };
    const r = await fetch('/api/newsletter/admin/dispatch', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify(body),
    });
    const d = await r.json();
    resultEl.className = 'result visible ' + (r.ok ? 'ok' : 'err');
    resultEl.textContent = JSON.stringify(d, null, 2);
  } catch (err) {
    resultEl.className = 'result visible err';
    resultEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Send';
  }
}

loadStats();
</script>
</body>
</html>`;
}

module.exports = {
  handleSubscribe,
  handleUnsubscribe,
  handleAdminStats,
  handleAdminDispatch,
  handleAdminPage,
};
