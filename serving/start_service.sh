#!/usr/bin/env bash
PORT=${PORT:-5000}
echo "=========================================================="
echo "  TRAINERMON DIAGNOSTIC SERVING LAUNCHER (Dual-Mode)"
echo "=========================================================="

DOCKER_AVAILABLE=0
if command -v docker >/dev/null 2>&1; then
    if docker version >/dev/null 2>&1; then
        DOCKER_AVAILABLE=1
    fi
fi

if [ "$DOCKER_AVAILABLE" -eq 1 ]; then
    echo "[Mode: Docker Container] Docker daemon online. Launching container on port ${PORT}..."
    docker compose -f docker-compose.yml up -d --build
    echo "Serving Container started on http://localhost:${PORT}"
else
    echo "[Mode: Native Python Flask] Docker daemon offline/stopped."
    echo "[AUTO-FALLBACK] Seamlessly launching production Flask inference engine natively on port ${PORT}..."
    python3 -m pip install -q -r requirements.txt
    PORT=${PORT} PYTHONUNBUFFERED=1 python3 app.py &
    echo "Flask process started on http://127.0.0.1:${PORT}"
fi
