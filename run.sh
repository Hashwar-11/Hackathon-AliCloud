#!/usr/bin/env bash
# =============================================================================
# ETD Platform — Unified Startup Script
# Usage:
#   ./run.sh          # Starts both FastAPI and Streamlit
#   ./run.sh api      # Starts FastAPI only (port 8000)
#   ./run.sh ui       # Starts Streamlit only (port 8501)
#   ./run.sh test     # Runs full pytest test suite
# =============================================================================

set -e

# Resolve Python environment
if [ -d "venv" ]; then
    PYTHON="venv/bin/python"
    STREAMLIT="venv/bin/streamlit"
    UVICORN="venv/bin/uvicorn"
    PYTEST="venv/bin/pytest"
else
    PYTHON="python3"
    STREAMLIT="streamlit"
    UVICORN="uvicorn"
    PYTEST="pytest"
fi

MODE="${1:-both}"

case "$MODE" in
    api)
        echo "Starting ETD FastAPI Service on http://0.0.0.0:8000 ..."
        exec $UVICORN src.api.app:app --host 0.0.0.0 --port 8000
        ;;
    ui)
        echo "Starting ETD Streamlit Control Room on http://0.0.0.0:8501 ..."
        exec $STREAMLIT run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    test)
        echo "Running ETD Test Suite..."
        exec $PYTEST tests/ -v
        ;;
    both)
        echo "==========================================================="
        echo " Starting ETD (Electricity Theft Detection) Platform"
        echo " Backend API:   http://localhost:8000 (Docs: /docs)"
        echo " Dashboard UI:  http://localhost:8501"
        echo "==========================================================="

        # Trap Ctrl+C to terminate both background jobs
        trap 'kill $(jobs -p) 2>/dev/null || true; exit' SIGINT SIGTERM EXIT

        $UVICORN src.api.app:app --host 0.0.0.0 --port 8000 &
        API_PID=$!

        # Wait briefly for API to initialize
        sleep 2

        $STREAMLIT run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 &
        UI_PID=$!

        wait $API_PID $UI_PID
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Usage: ./run.sh [both|api|ui|test]"
        exit 1
        ;;
esac
