/**
 * telemetry.js — WebSocket client for the Cesium nanosatellite dashboard.
 *
 * Handles two message formats from the Python OutputRouter:
 *
 *   1. cesium_telemetry  (preferred, added in Phase 7):
 *        { "type": "cesium_telemetry", "latitude_deg": …, "mcu_alert": …, … }
 *
 *   2. OpenMCT per-field envelope (legacy fallback):
 *        { "id": "V_batt", "timestamp": …, "value": 3.85 }
 *      String fields (mcu_alert, anomaly_type) are inferred from thresholds
 *      when only the legacy format is available.
 *
 * Requires:  window.satellite       (set by satellite.js)
 *            window._setInjectButtonState  (set by index.html)
 */
(function (global) {
  'use strict';

  // ── Codespaces-aware WebSocket URL ───────────────────────────────────────
  function getWebSocketURL() {
    if (window.location.hostname === 'localhost' ||
        window.location.hostname === '127.0.0.1') {
      return 'ws://localhost:8765';
    }
    // In Codespaces: page is on -8081 subdomain, WebSocket is on -8765 subdomain
    var host = window.location.hostname.replace('-8081', '-8765');
    return 'wss://' + host;
  }

  // ── UI helpers ────────────────────────────────────────────────────────────
  function setConnectionState(state) {
    var dot  = document.getElementById('connDot');
    var text = document.getElementById('connText');
    if (!dot || !text) return;
    dot.className = 'conn-dot';
    switch (state) {
      case 'connected':
        dot.classList.add('dot-connected');
        text.textContent = 'CONNECTED';
        break;
      case 'disconnected':
        dot.classList.add('dot-disconnected');
        text.textContent = 'DISCONNECTED';
        break;
      default:
        dot.classList.add('dot-connecting');
        text.textContent = 'CONNECTING...';
    }
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function fmt(val, digits) {
    var n = parseFloat(val);
    return isNaN(n) ? '—' : n.toFixed(digits);
  }

  // ── Apply a complete telemetry snapshot to the UI and 3-D globe ──────────
  function applyTelemetry(t) {
    var lat   = parseFloat(t.latitude_deg);
    var lon   = parseFloat(t.longitude_deg);
    var alt   = parseFloat(t.altitude_km);
    var level = ((t.mcu_alert || 'NORMAL') + '').toUpperCase();
    var atype = (t.anomaly_type || 'normal') + '';

    // ── 3-D satellite ────────────────────────────────────────────────────
    if (!isNaN(lat) && !isNaN(lon) && !isNaN(alt) && global.satellite) {
      global.satellite.updateSatellitePosition(lat, lon, alt);
      global.satellite.addOrbitTrailPoint(lat, lon, alt);
    }
    if (global.satellite) {
      global.satellite.updateAnomalyState(level, atype);
    }

    // ── Status panel ──────────────────────────────────────────────────────
    var badge = document.getElementById('alertBadge');
    if (badge) {
      badge.textContent = level;
      badge.className   = '';
      if      (level === 'CRITICAL') badge.className = 'badge-critical';
      else if (level === 'WARNING')  badge.className = 'badge-warning';
      else                           badge.className = 'badge-normal';
    }

    setText('vBatt',      fmt(t.V_batt, 1));
    setText('tBatt',      fmt(t.T_batt, 1));
    setText('soc',        fmt(t.SOC,    0));
    setText('anomalyType',
      (atype === 'normal' || !atype) ? 'none' : atype.replace(/_/g, ' '));
    setText('lastUpdate',
      new Date().toLocaleTimeString('en-GB', { hour12: false }));

    // ── Flash border on anomaly ───────────────────────────────────────────
    if (t.anomaly_active) {
      var panel = document.getElementById('statusPanel');
      if (panel) {
        panel.classList.remove('flash-warning', 'flash-critical');
        void panel.offsetWidth; // force reflow to restart animation
        panel.classList.add(level === 'CRITICAL' ? 'flash-critical' : 'flash-warning');
      }
      console.log('[ANOMALY] type=' + atype + ' alert=' + level + ' t=' + (t.timestamp || ''));
    }

    // ── Inject button state ───────────────────────────────────────────────
    if (global._setInjectButtonState) global._setInjectButtonState(level);
  }

  // ── OpenMCT per-field envelope buffer ─────────────────────────────────────
  // OpenMCT sends one JSON per numeric field.  We batch them over 200 ms
  // then fire a single UI update — avoids a partial update on each packet.
  var fieldBuffer  = {};
  var flushTimer   = null;
  var KEY_TRIO     = ['latitude_deg', 'longitude_deg', 'altitude_km'];

  function onFieldEnvelope(id, value) {
    fieldBuffer[id] = value;

    // Eager flush once we have all three position fields
    var hasPosition = KEY_TRIO.every(function (k) {
      return fieldBuffer[k] !== undefined;
    });
    if (hasPosition) {
      if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
      flushBuffer();
      return;
    }

    // Debounced flush for any other field
    if (flushTimer) clearTimeout(flushTimer);
    flushTimer = setTimeout(flushBuffer, 200);
  }

  function flushBuffer() {
    flushTimer = null;
    if (Object.keys(fieldBuffer).length === 0) return;

    var snap = Object.assign({}, fieldBuffer);

    // Infer string fields from numeric thresholds when not present
    if (!snap.mcu_alert) {
      var v = snap.V_batt, i = snap.I_batt, t = snap.T_batt;
      if ((t !== undefined && t > 60) ||
          (v !== undefined && v < 3.2) ||
          (i !== undefined && i > 3.0)) {
        snap.mcu_alert     = 'CRITICAL';
        snap.anomaly_active = true;
        snap.anomaly_type   = 'detected';
      } else if ((t !== undefined && t > 50) || (i !== undefined && i > 2.5)) {
        snap.mcu_alert     = 'WARNING';
        snap.anomaly_active = true;
        snap.anomaly_type   = 'detected';
      } else {
        snap.mcu_alert     = 'NORMAL';
        snap.anomaly_active = false;
        snap.anomaly_type   = 'normal';
      }
    }

    applyTelemetry(snap);
  }

  // ── WebSocket lifecycle ───────────────────────────────────────────────────
  var ws               = null;
  var reconnectTimer   = null;

  function connect() {
    setConnectionState('connecting');
    var url = getWebSocketURL();

    try {
      ws = new WebSocket(url);
    } catch (e) {
      setConnectionState('disconnected');
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      setConnectionState('connected');
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = function (event) {
      var msg;
      try { msg = JSON.parse(event.data); } catch (e) { return; }

      if (msg.type === 'cesium_telemetry') {
        // Full snapshot — direct path, fastest UI update
        applyTelemetry(msg);
      } else if (msg.id !== undefined && msg.value !== undefined) {
        // OpenMCT per-field envelope — buffer and batch
        onFieldEnvelope(msg.id, msg.value);
      }
    };

    ws.onerror = function () {
      setConnectionState('disconnected');
    };

    ws.onclose = function () {
      setConnectionState('disconnected');
      ws = null;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, 3000);
  }

  // Wait one tick so satellite.js has fully set up window.satellite
  setTimeout(connect, 100);

})(window);
