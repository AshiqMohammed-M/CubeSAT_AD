/**
 * EPSTelemetryPlugin
 * ------------------
 * OpenMCT plugin that registers all EPS Guardian telemetry fields as
 * real-time measurements, grouped into logical sub-folders in the object tree.
 *
 * Usage (in index.html):
 *   openmct.install(EPSTelemetryPlugin({
 *     wsUrl:     'ws://localhost:8080',
 *     schemaUrl: '/api/telemetry/schema',
 *   }));
 *
 * Message format expected from the WebSocket (one per field per step):
 *   { "id": "<field_key>", "timestamp": <unix_ms>, "value": <number|string> }
 */

/* global openmct */

'use strict';

// ---------------------------------------------------------------------------
// Namespace used for all EPS domain objects
// ---------------------------------------------------------------------------
const EPS_NAMESPACE = 'eps.guardian';

// ---------------------------------------------------------------------------
// Group definitions  (order matters – shown as-is in the object tree)
// ---------------------------------------------------------------------------
const GROUPS = [
  { key: 'battery', name: 'Battery',     keys: ['V_batt', 'I_batt', 'T_batt', 'P_batt', 'SOC'] },
  { key: 'solar',   name: 'Solar Array', keys: ['V_solar', 'I_solar', 'P_solar'] },
  { key: 'bus',     name: 'Power Bus',   keys: ['V_bus', 'I_bus', 'P_bus', 'converter_ratio', 'power_balance'] },
  { key: 'system',  name: 'System',      keys: ['T_eps', 'altitude_km', 'latitude_deg', 'longitude_deg', 'mission_time_s'] },
  { key: 'alerts',  name: 'Alerts',      keys: ['anomaly_type', 'mcu_alert', 'anomaly_active', 'orbit_sunlight'] },
];

// Root folder object identifier
const ROOT_KEY  = 'eps-root';
const ROOT_ID   = { namespace: EPS_NAMESPACE, key: ROOT_KEY };

// ---------------------------------------------------------------------------
// Helper: build a domain-object id from a field/group key
// ---------------------------------------------------------------------------
function mkId(key) {
  return { namespace: EPS_NAMESPACE, key };
}

function mkKeyStr(id) {
  return `${id.namespace}:${id.key}`;
}

// ---------------------------------------------------------------------------
// EPSTelemetryPlugin factory
// ---------------------------------------------------------------------------
window.EPSTelemetryPlugin = function EPSTelemetryPlugin({ wsUrl, schemaUrl }) {

  return function install(openmct) {

    // ── Step 1: fetch schema from REST endpoint ───────────────────────────
    let schemaCache = null;

    async function getSchema() {
      if (schemaCache) return schemaCache;
      const resp = await fetch(schemaUrl);
      if (!resp.ok) throw new Error(`Schema fetch failed: ${resp.status}`);
      schemaCache = await resp.json();
      return schemaCache;
    }

    // ── Step 2: WebSocket subscription manager ────────────────────────────
    let ws          = null;
    let wsReady     = false;
    // Map of fieldKey → Set of OpenMCT callback functions
    const listeners = new Map();
    // In-memory history buffer: fieldKey → Array of {id, timestamp, value}
    // Capped at MAX_HISTORY points per field to bound memory.
    const history   = new Map();
    const MAX_HISTORY = 3600; // ~1 hour at 1 Hz

    function addToHistory(fieldKey, datum) {
      if (!history.has(fieldKey)) history.set(fieldKey, []);
      const buf = history.get(fieldKey);
      buf.push(datum);
      if (buf.length > MAX_HISTORY) buf.shift();
    }

    function connectWs() {
      ws = new WebSocket(wsUrl);

      ws.addEventListener('open', () => {
        wsReady = true;
        console.log('[EPS plugin] WebSocket connected to bridge server');
      });

      ws.addEventListener('message', (event) => {
        let msg;
        try { msg = JSON.parse(event.data); } catch { return; }

        const { id, timestamp, value } = msg;
        if (!id) return;

        // Deliver datum to all subscribers for this field
        const datum = {
          id:        mkKeyStr(mkId(id)),
          timestamp, // unix ms
          value,
        };

        addToHistory(id, datum);

        const cbs = listeners.get(id);
        if (!cbs || cbs.size === 0) return;
        cbs.forEach((cb) => { try { cb(datum); } catch { /* ignore */ } });
      });

      ws.addEventListener('close', () => {
        wsReady = false;
        console.warn('[EPS plugin] Bridge WebSocket closed – reconnecting in 3 s');
        setTimeout(connectWs, 3000);
      });

      ws.addEventListener('error', (err) => {
        console.error('[EPS plugin] WebSocket error:', err);
      });
    }

    connectWs();

    // ── Step 3: Object Provider ───────────────────────────────────────────
    // Provides root folder, group folders, and leaf measurement objects.

    openmct.objects.addProvider(EPS_NAMESPACE, {
      get: async function (identifier) {
        const schema = await getSchema();
        const measurements = schema.measurements || {};
        const key = identifier.key;

        // Root folder
        if (key === ROOT_KEY) {
          return {
            identifier: ROOT_ID,
            name: 'EPS Guardian',
            type: 'folder',
            location: 'ROOT',
            composition: GROUPS.map((g) => mkId(g.key)),
          };
        }

        // Group folder
        const group = GROUPS.find((g) => g.key === key);
        if (group) {
          return {
            identifier: mkId(group.key),
            name: group.name,
            type: 'folder',
            location: mkKeyStr(ROOT_ID),
            composition: group.keys
              .filter((k) => measurements[k])
              .map((k) => mkId(k)),
          };
        }

        // Leaf measurement
        const meta = measurements[key];
        if (!meta) return undefined;

        const parentGroup = GROUPS.find((g) => g.keys.includes(key));

        return {
          identifier: mkId(key),
          name:       meta.name || key,
          type:       'eps.telemetry',
          location:   parentGroup
            ? mkKeyStr(mkId(parentGroup.key))
            : mkKeyStr(ROOT_ID),
          telemetry: {
            values: [
              {
                key:       'value',
                name:      meta.name || key,
                unit:      meta.unit || '',
                format:    meta.format || 'float',
                min:       meta.min,
                max:       meta.max,
                hints:     { range: 1 },
              },
              {
                key:    'utc',
                source: 'timestamp',
                name:   'Timestamp',
                format: 'utc',
                hints:  { domain: 1 },
              },
            ],
          },
        };
      },
    });

    // ── Step 4: Type definitions ──────────────────────────────────────────

    openmct.types.addType('eps.telemetry', {
      name:        'EPS Telemetry Point',
      description: 'A real-time EPS Guardian sensor measurement',
      cssClass:    'icon-telemetry',
    });

    // ── Step 5: Composition Provider ─────────────────────────────────────
    // Tells OpenMCT how to expand folders.

    openmct.composition.addProvider({
      appliesTo(domainObject) {
        return (
          domainObject.type === 'folder' &&
          domainObject.identifier.namespace === EPS_NAMESPACE
        );
      },
      load(domainObject) {
        return Promise.resolve(domainObject.composition || []);
      },
    });

    // ── Step 6: Telemetry Provider ────────────────────────────────────────

    openmct.telemetry.addProvider({
      supportsSubscribe(domainObject) {
        return domainObject.type === 'eps.telemetry';
      },

      subscribe(domainObject, callback) {
        const fieldKey = domainObject.identifier.key;
        if (!listeners.has(fieldKey)) listeners.set(fieldKey, new Set());
        listeners.get(fieldKey).add(callback);

        // Unsubscribe function
        return function unsubscribe() {
          const cbs = listeners.get(fieldKey);
          if (cbs) cbs.delete(callback);
        };
      },

      supportsRequest(domainObject) {
        return domainObject.type === 'eps.telemetry';
      },

      request(domainObject, options) {
        const fieldKey = domainObject.identifier.key;
        const buf = history.get(fieldKey) || [];
        const start = options && options.start ? options.start : 0;
        const end   = options && options.end   ? options.end   : Date.now();
        return Promise.resolve(buf.filter((d) => d.timestamp >= start && d.timestamp <= end));
      },
    });

    // ── Step 7: Register root namespace in ROOT composition ───────────────

    openmct.objects.addRoot(ROOT_ID);
  };
};
