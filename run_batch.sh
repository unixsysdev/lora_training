#!/usr/bin/env bash
# Run a batch of prompts for testing
set -euo pipefail

source .venv/bin/activate

# Default values
PROMPT_FILE=""
TOPIC="general"
OUTPUT=""
DELAY=0.5
MAX_TOKENS=512

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --file|-f)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --topic|-t)
      TOPIC="$2"
      shift 2
      ;;
    --output|-o)
      OUTPUT="$2"
      shift 2
      ;;
    --delay|-d)
      DELAY="$2"
      shift 2
      ;;
    --max-tokens|-m)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: ./run_batch.sh [OPTIONS]"
      echo "Options:"
      echo "  --file, -f FILE       Prompts file (required)"
      echo "  --topic, -t TOPIC     Topic label (default: general)"
      echo "  --output, -o FILE     Output file (default: results_TIMESTAMP.jsonl)"
      echo "  --delay, -d SECONDS   Delay between requests (default: 0.5)"
      echo "  --max-tokens, -m NUM  Max tokens to generate (default: 512)"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$PROMPT_FILE" ]]; then
  echo "Error: Prompts file is required"
  echo "Use --help for usage information"
  exit 1
fi

# Set default output file if not specified
if [[ -z "$OUTPUT" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  OUTPUT="results_${TIMESTAMP}.jsonl"
fi

# Run the batch
echo "Running batch test with:"
echo "  File: $PROMPT_FILE"
echo "  Topic: $TOPIC"
echo "  Output: $OUTPUT"
echo "  Delay: $DELAY seconds"
echo "  Max tokens: $MAX_TOKENS"
echo

python batch_test.py \
  --topic "$TOPIC" \
  --output "$OUTPUT" \
  --delay "$DELAY" \
  --max-tokens "$MAX_TOKENS" \
  "$PROMPT_FILE"
