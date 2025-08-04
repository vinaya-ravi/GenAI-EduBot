#!/bin/bash

# Build the container
podman build -t gemma-agent .

# Run the container
podman run -d \
  --name gemma-agent \
  -p 8000:8000 \
  -e MODEL_ID=google/gemma-3-27b-it \
  -e INFERENCE_SERVER_URL=http://vllm-server:5000/v1 \
  -e MAX_RETRIES=3 \
  -e RETRY_DELAY=2 \
  -e REQUEST_TIMEOUT=30 \
  -e CHAINLIT_HOST=0.0.0.0 \
  -e CHAINLIT_PORT=8000 \
  -e LOG_LEVEL=INFO \
  gemma-agent 