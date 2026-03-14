/**
 * EPS Guardian – OpenMCT Bridge Server
 *
 * Responsibilities
 * ----------------
 * 1. Serve the OpenMCT web application as static files on port 8080.
 * 2. Bridge the Python telemetry WebSocket (upstream, ws://localhost:8765)
 *    to every connected browser client (downstream).
 * 3. Auto-reconnect to the Python server when the connection drops.
 * 4. Expose GET /api/telemetry/schema returning the EPS field schema.
 *
 * Start:  node server.js
 * Open:   http://localhost:8080
 */

'use strict';

const express = require('express');
const http    = require('http');
const path    = require('path');
const fs      = require('fs');
const WebSocket = require('ws');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const HTTP_PORT          = process.env.PORT          || 8080;
const PYTHON_WS_URL      = process.env.PYTHON_WS_URL || 'ws://localhost:8765';
const RECONNECT_DELAY_MS = 3000;   // wait before reconnecting to Python server
const SCHEMA_PATH        = path.join(__dirname, 'schema.json');

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------
const app    = express();
const server = http.createServer(app);

// ── CORS — required for Codespaces port-forwarded URLs ──────────────────────
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin',  '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// Serve OpenMCT from node_modules
const openmctPath = path.join(__dirname, 'node_modules', 'openmct', 'dist');
if (fs.existsSync(openmctPath)) {
  app.use('/openmct', express.static(openmctPath));
} else {
  console.warn('[server] OpenMCT dist not found at', openmctPath,
    '— run "npm install" first');
}

// Serve our plugin and index from the client folder
const clientPath = path.join(__dirname, '..', 'openmct_client');
app.use('/client', express.static(clientPath));

// Serve index.html from this directory
app.use(express.static(__dirname));

// REST: EPS telemetry schema
app.get('/api/telemetry/schema', (req, res) => {
  try {
    const schema = JSON.parse(fs.readFileSync(SCHEMA_PATH, 'utf8'));
    res.json(schema);
  } catch (err) {
    console.error('[api] Failed to read schema:', err.message);
    res.status(500).json({ error: 'Schema file not available' });
  }
});

// REST: health-check
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    upstreamConnected: upstreamConnected,
    downstreamClients: downstreamClients.size,
  });
});

// ---------------------------------------------------------------------------
// Downstream WebSocket server  (browser → this server)
// ---------------------------------------------------------------------------
const wss = new WebSocket.Server({ server });
const downstreamClients = new Set();

wss.on('connection', (ws, req) => {
  const ip = req.socket.remoteAddress;
  console.log(`[downstream] Client connected  (${downstreamClients.size + 1} total)  ip=${ip}`);
  downstreamClients.add(ws);

  ws.on('close', () => {
    downstreamClients.delete(ws);
    console.log(`[downstream] Client disconnected (${downstreamClients.size} remaining)`);
  });

  ws.on('error', (err) => {
    console.error('[downstream] Client error:', err.message);
    downstreamClients.delete(ws);
  });
});

/**
 * Broadcast a raw message string to all connected browser clients.
 * Dead sockets are removed silently.
 */
function broadcast(message) {
  const dead = [];
  for (const client of downstreamClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message, (err) => {
        if (err) dead.push(client);
      });
    } else {
      dead.push(client);
    }
  }
  dead.forEach((c) => downstreamClients.delete(c));
}

// ---------------------------------------------------------------------------
// Upstream WebSocket client  (this server → Python telemetry server)
// ---------------------------------------------------------------------------
let upstreamWs         = null;
let upstreamConnected  = false;
let reconnectTimer     = null;

function connectUpstream() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  console.log(`[upstream] Connecting to Python server: ${PYTHON_WS_URL}`);
  upstreamWs = new WebSocket(PYTHON_WS_URL);

  upstreamWs.on('open', () => {
    upstreamConnected = true;
    console.log('[upstream] Connected to Python WebSocket server');
  });

  // Every message from Python is forwarded verbatim to all browser clients
  upstreamWs.on('message', (data) => {
    const msg = data.toString();
    if (downstreamClients.size > 0) {
      broadcast(msg);
    }
  });

  upstreamWs.on('close', (code, reason) => {
    upstreamConnected = false;
    console.warn(
      `[upstream] Disconnected (code=${code} reason="${reason}") ` +
      `— reconnecting in ${RECONNECT_DELAY_MS}ms`
    );
    scheduleReconnect();
  });

  upstreamWs.on('error', (err) => {
    // 'close' event fires after 'error', so we only log here
    if (err.code === 'ECONNREFUSED') {
      console.warn(`[upstream] Python server not reachable (${PYTHON_WS_URL}) — will retry`);
    } else {
      console.error('[upstream] Error:', err.message);
    }
  });
}

function scheduleReconnect() {
  if (!reconnectTimer) {
    reconnectTimer = setTimeout(connectUpstream, RECONNECT_DELAY_MS);
  }
}

// ---------------------------------------------------------------------------
// index.html  (generated so no separate file is required)
// ---------------------------------------------------------------------------
app.get('/', (_req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>EPS Guardian – OpenMCT</title>
  <link rel="stylesheet" href="/openmct/espressoTheme.css" />
  <script src="/openmct/openmct.js"></script>
</head>
<body>
  <div id="app"></div>
  <script src="/client/eps_telemetry_plugin.js"></script>
  <script>
    // ---- Bootstrap OpenMCT ----
    const openmct = window.openmct;

    openmct.setAssetPath('/openmct');

    // Built-in plugins
    openmct.install(openmct.plugins.LocalStorage());
    openmct.install(openmct.plugins.Espresso());
    openmct.install(openmct.plugins.Timeline());
    openmct.install(openmct.plugins.UTCTimeSystem());
    openmct.install(openmct.plugins.Clock({ enableClockIndicator: true }));
    openmct.install(openmct.plugins.LADTable());
    openmct.install(openmct.plugins.Conductor({
      menuOptions: [
        {
          name: 'Real-time',
          timeSystem: 'utc',
          clock: 'local',
          clockOffsets: { start: -30 * 60 * 1000, end: 30 * 1000 },
        },
        {
          name: 'Fixed',
          timeSystem: 'utc',
          bounds: {
            start: Date.now() - 2 * 60 * 60 * 1000,
            end:   Date.now(),
          },
        },
      ],
    }));

    // EPS Guardian plugin
    // Use wss:// when served over HTTPS (Codespaces), ws:// for localhost
    openmct.install(EPSTelemetryPlugin({
      wsUrl:     (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host,
      schemaUrl: '/api/telemetry/schema',
    }));

    openmct.start();
  </script>
</body>
</html>`);
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
server.listen(HTTP_PORT, () => {
  console.log(`[server] EPS Guardian OpenMCT listening on http://localhost:${HTTP_PORT}`);
  connectUpstream();
});

// Graceful shutdown
process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

function shutdown(signal) {
  console.log(`[server] ${signal} received – shutting down`);
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (upstreamWs)     upstreamWs.terminate();
  server.close(() => process.exit(0));
}
