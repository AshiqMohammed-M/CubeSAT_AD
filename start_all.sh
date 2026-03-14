#!/bin/bash
# start_all.sh — Launch the full nanosatellite pipeline in one command.
#
# Services started:
#   8080  OpenMCT telemetry graphs  (Node bridge → Python WS)
#   8081  Cesium 3-D orbit view     (direct → Python WS on 8765)
#   8765  Python WebSocket server   (OutputRouter broadcast)
#
# Usage:
#   bash start_all.sh
#
# In Codespaces: forward ports 8080, 8081, and 8765 as PUBLIC (see PORTS.md).

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# Verify venv exists
if [ ! -f "${PYTHON}" ]; then
  echo "[ERROR] Python venv not found at ${PYTHON}"
  echo "        Run:  python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo ""
echo "========================================"
echo "  Nanosatellite EPS Guardian — startup"
echo "========================================"
echo ""

# ── [1/3] OpenMCT dashboard (port 8080) ──────────────────────────────────────
echo "[1/3] Starting OpenMCT dashboard on port 8080..."
cd "${PROJECT_ROOT}/dashboard/openmct_server"
if [ ! -d node_modules ]; then
  echo "      Installing npm packages..."
  npm install --silent
fi
node server.js > /tmp/openmct.log 2>&1 &
OPENMCT_PID=$!

# ── [2/3] Cesium dashboard (port 8081) ───────────────────────────────────────
echo "[2/3] Starting Cesium 3-D dashboard on port 8081..."
cd "${PROJECT_ROOT}/dashboard/cesium"
if [ ! -d node_modules ]; then
  echo "      Installing npm packages..."
  npm install --silent
fi
node server.js > /tmp/cesium.log 2>&1 &
CESIUM_PID=$!

# ── [3/3] Python pipeline (WebSocket on 8765) ────────────────────────────────
echo "[3/3] Starting Python telemetry pipeline (WS on 8765)..."
cd "${PROJECT_ROOT}"
"${PYTHON}" src/main.py --mode realtime --no-serial --no-influx > /tmp/pipeline.log 2>&1 &
PYTHON_PID=$!

# Give everything a moment to bind ports
sleep 2

echo ""
echo "  ✓ All services started"
echo ""
echo "  OpenMCT  → http://localhost:8080   (telemetry plots)"
echo "  Cesium   → http://localhost:8081   (3-D orbit view)"
echo "  WS data  → ws://localhost:8765     (Python source)"
echo ""
echo "  Logs:"
echo "    /tmp/pipeline.log  (Python)"
echo "    /tmp/openmct.log   (OpenMCT)"
echo "    /tmp/cesium.log    (Cesium)"
echo ""
echo "  In Codespaces: forward ports 8080, 8081, 8765 as PUBLIC (see PORTS.md)"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Stop all child processes on Ctrl+C / script exit
cleanup() {
  echo ""
  echo "Stopping all services..."
  kill "${OPENMCT_PID}" "${CESIUM_PID}" "${PYTHON_PID}" 2>/dev/null || true
  wait "${OPENMCT_PID}" "${CESIUM_PID}" "${PYTHON_PID}" 2>/dev/null || true
  echo "Done."
  exit 0
}
trap cleanup INT TERM

wait
