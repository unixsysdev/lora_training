#!/usr/bin/env bash
# Monitor logs, metrics, and system status
set -euo pipefail

source .venv/bin/activate

# Parse arguments
MODE="logs"
if [[ $# -gt 0 ]]; then
  MODE="$1"
fi

case "$MODE" in
  logs)
    echo "Monitoring logs (Ctrl+C to exit)..."
    tail -f logs/app.log
    ;;
  metrics)
    echo "Displaying metrics..."
    if [[ -f "logs/metrics.tsv" ]]; then
      echo "Current metrics:"
      column -t -s \t' logs/metrics.tsv'
    else
      echo "No metrics file found."
    fi
    ;;
  health)
    echo "Running health check..."
    python health_check.py
    ;;
  responses)
    echo "Recent teacher responses:"
    if [[ -d "responses" ]]; then
      ls -lt responses | head -n 10
      echo
      echo "To view a specific response file:"
      echo "  cat responses/FILENAME"
    else
      echo "No responses directory found."
    fi
    ;;
  *)
    echo "Usage: ./monitor.sh [logs|metrics|health|responses]"
    echo "  logs      - Show live logs (default)"
    echo "  metrics   - Display learning metrics"
    echo "  health    - Run system health check"
    echo "  responses - List recent teacher responses"
    ;;
esac
