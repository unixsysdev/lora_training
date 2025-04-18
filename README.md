# Continuous Learning Lab

A professional implementation of a continuous learning system using LoRA fine-tuning with a teacher-student architecture.

## Overview

This system implements a continuous learning pipeline where:

1. A student model (quantized for efficiency) serves responses
2. Teacher models (via OpenRouter) evaluate and improve responses
3. A LoRA adapter is continuously updated based on feedback
4. All interactions and raw responses are logged for analysis

## Requirements

- Python 3.8+ with venv support
- CUDA-compatible GPU (recommended)
- Redis server

## Configuration

Edit the `.env` file to configure:

- Student model selection and quantization
- Teacher models via OpenRouter
- LoRA and learning parameters
- System parameters

## Directory Structure

```
continuous_learning_lab/
├── .env                 # Environment configuration
├── requirements.txt     # Python dependencies
├── src_logger.py        # Logging utilities
├── src_model.py         # Model loading and inference
├── src_server.py        # FastAPI server
├── src_worker.py        # Celery worker for evaluation
├── health_check.py      # System health check utility
├── batch_test.py        # Batch testing utility
├── start.sh             # Startup script
├── monitor.sh           # Monitoring script
├── run_batch.sh         # Batch execution script
├── logs/                # Log files
├── data/                # Data files
├── models/              # Model files
├── batches/             # Batch test files
└── responses/           # Raw teacher responses
```

## Getting Started

1. Edit the `.env` file and add your OpenRouter API key
2. Start the system: `./start.sh`
3. Monitor logs: `./monitor.sh logs`
4. Check system health: `./monitor.sh health`
5. Run a batch test: `./run_batch.sh --file batches/reasoning_test.txt --topic reasoning`

## API Endpoints

- `GET /`: Basic system info
- `POST /chat`: Generate a response and queue for evaluation
  - Request body:
    ```json
    {
      "prompt": "Your query here",
      "topic": "category",
      "max_new_tokens": 512,
      "temperature": 0.7
    }
    ```

## Monitoring and Analysis

- View logs: `./monitor.sh logs`
- Check metrics: `./monitor.sh metrics`
- View teacher responses: `./monitor.sh responses`
- Run health check: `./monitor.sh health`

## Performance Tuning

Adjust the following parameters in `.env` for performance:

- `QUANT_BITS`: Lower for less memory usage, higher for better quality
- `TRAIN_BATCH_SIZE`: Increase for faster training if memory allows
- `THRESHOLD_START`: Initial quality threshold (0-10)
- `GOOD_BATCH_SIZE`: Number of examples before batch update

## License

This project is licensed under the MIT License - see the LICENSE file for details.
