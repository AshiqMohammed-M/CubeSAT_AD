/**
 * server.js — Express static-file server for the Cesium dashboard.
 *
 * Serves dashboard/cesium/ on port 8080 with full CORS headers so the page
 * works when the Codespaces Ports panel forwards it to a public URL.
 *
 * Usage:
 *   npm start                 (from dashboard/cesium/)
 *   node server.js
 */
'use strict';

const express = require('express');
const path    = require('path');

const app  = express();
const PORT = 8080;

// ── CORS — required for Codespaces port-forwarded URLs ──────────────────────
app.use(function (req, res, next) {
  res.header('Access-Control-Allow-Origin',  '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// ── Static files (index.html, satellite.js, telemetry.js, …) ────────────────
app.use(express.static(path.join(__dirname)));

// ── API endpoints ────────────────────────────────────────────────────────────
app.get('/health', function (req, res) {
  res.json({ status: 'ok', timestamp: Date.now() });
});

app.get('/config', function (req, res) {
  res.json({ websocket_port: 8765, version: '1.0' });
});

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, function () {
  console.log('');
  console.log('  Cesium dashboard : http://localhost:' + PORT);
  console.log('  WebSocket bridge : ws://localhost:8765');
  console.log('  In Codespaces    : forward both ports 8080 and 8765');
  console.log('  Set ports to PUBLIC visibility in the Ports panel');
  console.log('');
});
