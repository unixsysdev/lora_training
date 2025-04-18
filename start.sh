#!/usr/bin/env bash
# Start all components of the Continuous Learning Lab
set -euo pipefail

echo "Starting Continuous Learning Lab..."
source .venv/bin/activate

# Check if Redis is running, start if not
redis-cli ping &>/dev/null || {
  echo "Starting Redis server..."
  redis-server --save "" --appendonly no --daemonize yes
  sleep 1
}

# Start API server
echo "Starting API server..."
python src_server.py &
SERVER_PID=$!
echo "API server started with PID $SERVER_PID"

# Give the server a moment to start
sleep 2

# Start Celery worker
echo "Starting Celery worker..."
celery -A src_worker worker --loglevel=INFO

# Cleanup on exit
trap "kill $SERVER_PID" EXIT
