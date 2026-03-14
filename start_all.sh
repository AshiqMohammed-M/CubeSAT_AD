#!/usr/bin/env bash
# start_all.sh — Launch the full nanosatellite pipeline in one command.
#
# Services started:
#   5000  Inject API endpoint      (FastAPI thread inside src/main.py)
#   8080  OpenMCT telemetry graphs  (Node bridge → Python WS)
#   8081  Cesium 3-D orbit view     (direct → Python WS on 8765)
#   8765  Python WebSocket server   (OutputRouter broadcast)
#
# Usage:
#   bash start_all.sh
#
# In Codespaces: forward ports 5000, 8080, 8081, and 8765 as PUBLIC.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
OPENMCT_DIR="${PROJECT_ROOT}/dashboard/openmct_server"
CESIUM_DIR="${PROJECT_ROOT}/dashboard/cesium"

OPENMCT_LOG="/tmp/openmct.log"
CESIUM_LOG="/tmp/cesium.log"
PYTHON_LOG="/tmp/pipeline.log"

# Tuneable startup timeouts (seconds). Override via environment variables.
PYTHON_START_TIMEOUT="${PYTHON_START_TIMEOUT:-90}"
SERVICE_START_TIMEOUT="${SERVICE_START_TIMEOUT:-30}"

OPENMCT_PID=""
CESIUM_PID=""
PYTHON_PID=""
CLEANUP_DONE=0

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command: ${cmd}"
    exit 1
  fi
}

release_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:${port}" || true)"
  if [ -n "${pids}" ]; then
    echo "      Releasing port ${port} (PID(s): $(echo "${pids}" | tr '\n' ' '))"
    kill ${pids} 2>/dev/null || true
    sleep 1
  fi
}

wait_for_port() {
  local port="$1"
  local label="$2"
  local timeout_s="${3:-20}"
  local elapsed=0
  while [ "${elapsed}" -lt "${timeout_s}" ]; do
    if lsof -ti "tcp:${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "[ERROR] ${label} did not bind port ${port} within ${timeout_s}s"
  return 1
}

ensure_node_modules() {
  local dir="$1"
  if [ ! -d "${dir}/node_modules" ]; then
    echo "      Installing npm packages in ${dir}..."
    (cd "${dir}" && npm install --silent)
  fi
}

cleanup() {
  local exit_code="$?"

  if [ "${CLEANUP_DONE}" -eq 1 ]; then
    return
  fi
  CLEANUP_DONE=1

  echo ""
  echo "Stopping all services..."

  if [ -n "${OPENMCT_PID}" ]; then
    kill "${OPENMCT_PID}" 2>/dev/null || true
    wait "${OPENMCT_PID}" 2>/dev/null || true
  fi

  if [ -n "${CESIUM_PID}" ]; then
    kill "${CESIUM_PID}" 2>/dev/null || true
    wait "${CESIUM_PID}" 2>/dev/null || true
  fi

  if [ -n "${PYTHON_PID}" ]; then
    kill "${PYTHON_PID}" 2>/dev/null || true
    wait "${PYTHON_PID}" 2>/dev/null || true
  fi

  echo "Done."

  if [ "${exit_code}" -ne 0 ]; then
    echo ""
    echo "Startup failed. Recent logs:"
    [ -f "${PYTHON_LOG}" ] && tail -n 20 "${PYTHON_LOG}" || true
    [ -f "${OPENMCT_LOG}" ] && tail -n 20 "${OPENMCT_LOG}" || true
    [ -f "${CESIUM_LOG}" ] && tail -n 20 "${CESIUM_LOG}" || true
  fi
}

trap cleanup EXIT INT TERM

# Verify venv exists
if [ ! -f "${PYTHON}" ]; then
  echo "[ERROR] Python venv not found at ${PYTHON}"
  echo "        Run:  python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

require_cmd lsof
require_cmd node
require_cmd npm

echo ""
echo "========================================"
echo "  Nanosatellite EPS Guardian — startup"
echo "========================================"
echo ""

echo "[preflight] Releasing required ports (5000, 8080, 8081, 8765)..."
release_port 5000
release_port 8080
release_port 8081
release_port 8765

echo "[preflight] Ensuring Node dependencies are installed..."
ensure_node_modules "${OPENMCT_DIR}"
ensure_node_modules "${CESIUM_DIR}"

# ── [1/3] Python pipeline (WebSocket on 8765 + API on 5000) ─────────────────
echo "[1/3] Starting Python telemetry pipeline (WS 8765 + API 5000)..."
cd "${PROJECT_ROOT}"
"${PYTHON}" src/main.py --mode realtime --no-serial --no-influx > "${PYTHON_LOG}" 2>&1 &
PYTHON_PID=$!

# Wait until both Python ports are available
wait_for_port 8765 "Python WebSocket" "${PYTHON_START_TIMEOUT}"
wait_for_port 5000 "Inject API" "${PYTHON_START_TIMEOUT}"

# ── [2/3] OpenMCT dashboard (port 8080) ─────────────────────────────────────
echo "[2/3] Starting OpenMCT dashboard on port 8080..."
cd "${OPENMCT_DIR}"
node server.js > "${OPENMCT_LOG}" 2>&1 &
OPENMCT_PID=$!
wait_for_port 8080 "OpenMCT dashboard" "${SERVICE_START_TIMEOUT}"

# ── [3/3] Cesium dashboard (port 8081) ──────────────────────────────────────
echo "[3/3] Starting Cesium 3-D dashboard on port 8081..."
cd "${CESIUM_DIR}"
node server.js > "${CESIUM_LOG}" 2>&1 &
CESIUM_PID=$!
wait_for_port 8081 "Cesium dashboard" "${SERVICE_START_TIMEOUT}"

echo ""
echo "  ✓ All services started"
echo ""
echo "  Inject API → http://localhost:5000   (POST /api/inject)"
echo "  OpenMCT  → http://localhost:8080   (telemetry plots)"
echo "  Cesium   → http://localhost:8081   (3-D orbit view)"
echo "  WS data  → ws://localhost:8765     (Python source)"
echo ""
echo "  Logs:"
echo "    ${PYTHON_LOG}  (Python)"
echo "    ${OPENMCT_LOG}   (OpenMCT)"
echo "    ${CESIUM_LOG}    (Cesium)"
echo ""
echo "  In Codespaces: forward ports 5000, 8080, 8081, 8765 as PUBLIC (see PORTS.md)"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

wait
