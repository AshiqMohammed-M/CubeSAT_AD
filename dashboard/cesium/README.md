# Cesium 3-D Satellite Dashboard

Real-time nanosatellite visualization — live orbit, anomaly state colouring,
and stakeholder-friendly status panel — powered by CesiumJS + the Python
`OutputRouter` WebSocket feed.

---

## One-time setup

```bash
# 1. Get a free Cesium ion token at https://cesium.com/ion
#    (enables "Earth at Night" dark imagery — optional but looks great)

# 2. Paste your token into index.html:
#    Cesium.Ion.defaultAccessToken = 'YOUR_CESIUM_TOKEN';

# 3. Install the Node dependency (only Express)
cd dashboard/cesium
npm install
```

---

## Start the dashboard

**Terminal 1 — Python telemetry pipeline**

```bash
# From the project root
python src/main.py --mode realtime --no-serial
```

**Terminal 2 — Cesium Node server**

```bash
cd dashboard/cesium
npm start
```

---

## Codespaces port setup (IMPORTANT)

Without public port visibility the WebSocket connection will be blocked.

1. Open the **Ports** panel in VS Code  
   *(bottom panel → Ports tab, or Ctrl+Shift+P → "Forward a Port")*
2. Forward **port 8081** → set Visibility to **Public**
3. Forward **port 8765** → set Visibility to **Public**
4. Click the 🌐 globe icon next to port **8081** to open the dashboard

---

## Verify it's working

| What to check | Expected |
|---------------|----------|
| Connection badge (top-right) | **CONNECTED** with green dot within 3 s |
| Globe | Earth renders, satellite label "SAT-01" appears |
| Status panel | Battery / temperature values updating each second |
| Inject button | Click → satellite turns red within 2 s |

---

## Troubleshooting

**`CesiumJS token error` in browser console**

```
Add your free token from cesium.com/ion to index.html
Cesium.Ion.defaultAccessToken = 'your_token_here';
```

The dashboard still works without a token — it uses bundled NaturalEarthII
imagery instead of the darker "Earth at Night" style.

**DISCONNECTED badge / no telemetry**

- Check the Python pipeline is running: `python src/main.py --mode realtime --no-serial`
- In Codespaces: confirm **both** ports 8080 and 8765 are set to **Public**
- The WebSocket URL is derived dynamically from the page hostname — no
  manual configuration needed.

**Satellite not visible on globe**

The satellite body is 2 m × 1 m × 0.5 m (true to CubeSat scale).  
Use the status panel camera fly-to (triggered automatically on anomaly state
changes) to zoom in, or scroll/pinch on the globe to zoom manually.
The coloured dot marker is always visible at any zoom level.

---

## File structure

```
dashboard/
└── cesium/
    ├── index.html    ← Full-screen CesiumJS globe + status overlay
    ├── satellite.js  ← Satellite entity, orbit trail, anomaly colouring
    ├── telemetry.js  ← WebSocket client + UI update logic
    ├── server.js     ← Express static server (port 8080)
    ├── package.json
    └── README.md
```
