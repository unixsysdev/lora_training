#!/usr/bin/env bash
# Initialize and test the Continuous Learning Lab
set -euo pipefail

# Activate virtual environment
source .venv/bin/activate

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p responses

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

# Test Redis
echo "Testing Redis connection..."
if ! redis-cli ping; then
  echo "Redis not running, starting..."
  redis-server --save "" --appendonly no --daemonize yes
  sleep 1
  if ! redis-cli ping; then
    echo "Error: Redis is not running properly."
    exit 1
  fi
fi

# Check if OpenRouter API key is set
source .env
if [[ "$OPENROUTER_API_KEY" == "your-api-key-here" ]]; then
  echo "⚠️  Warning: OpenRouter API key is not set."
  echo "    Please edit the .env file and set your OPENROUTER_API_KEY."
fi

# Run health check
echo "Setup complete! Running initial health check..."
python health_check.py

echo
echo "🎉 Continuous Learning Lab is ready!"
echo "    Start the system:  ./start.sh"
echo "    Monitor logs:      ./monitor.sh logs"
echo "    Run a batch test:  ./run_batch.sh --file batches/reasoning_test.txt"
echo
