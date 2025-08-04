#!/bin/bash

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "Podman is not installed. Please install it first."
    exit 1
fi

# Check for NVIDIA GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA SMI not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Define custom runtime path (adjust if necessary)
CRUN_RUNTIME="/home/${USER}/l/bin/crun"
if [ ! -x "${CRUN_RUNTIME}" ]; then
    echo "Custom runtime ${CRUN_RUNTIME} not found or not executable." 
    # Optionally, try finding crun in standard paths
    # CRUN_RUNTIME=$(command -v crun)
    # if [ -z "${CRUN_RUNTIME}" ]; then
    #    echo "Could not find crun runtime anywhere. Exiting."
    #    exit 1
    # fi
    # echo "Using crun found at: ${CRUN_RUNTIME}"
    # For now, we exit if the specified one isn't found
    exit 1
fi

# Create necessary directories
echo "Creating required directories..."
mkdir -p models/FAISS_INGEST/vectorstore
mkdir -p logs

# Clean up existing containers and pod
echo "Cleaning up existing containers and pod..."
podman pod rm -f app-pod || true
podman rmi -f localhost/vllm-server localhost/chainlit-app || true

# Create pod
echo "Creating pod for the application..."
podman pod create --name app-pod -p 5000:5000 -p 8000:8000

# Build images
echo "Building vLLM server image..."
DOCKER_BUILDKIT=1 podman build --no-cache -t localhost/vllm-server:latest -f Containerfile.vllm . || { echo "vLLM Image Build Failed!"; exit 1; }

echo "Building Chainlit app image..."
DOCKER_BUILDKIT=1 podman build --no-cache -t localhost/chainlit-app:latest -f Containerfile.chainlit . || { echo "Chainlit Image Build Failed!"; exit 1; }

# Start vLLM server
echo "Starting vLLM server..."
podman run \
    --runtime="${CRUN_RUNTIME}" \
    --device=nvidia.com/gpu=0 \
    --device=nvidia.com/gpu=1 \
    --device=nvidia.com/gpu=2 \
    --device=nvidia.com/gpu=3 \
    -d \
    --pod app-pod \
    --name vllm-server \
    --security-opt=label=disable \
    -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "${PWD}/models:/home/models:Z" \
    localhost/vllm-server:latest

# Wait for vLLM server to start
echo "Waiting for vLLM server to initialize..."
sleep 15

# Start Chainlit app
echo "Starting Chainlit app..."
podman run -d \
    --pod app-pod \
    --name chainlit-app \
    --security-opt=label=disable \
    -v "${PWD}/models:/home/models:Z" \
    -e VECTOR_DB_PATH=/home/models/FAISS_INGEST/vectorstore/db_faiss \
    localhost/chainlit-app:latest

echo "Application started!"
echo "vLLM server is accessible at: http://localhost:5000"
echo "Chainlit UI is accessible at: http://localhost:8000"
echo ""
echo "To check container status:"
echo "  podman ps"
echo ""
echo "To view logs:"
echo "  podman logs -f vllm-server"
echo "  podman logs -f chainlit-app"
echo ""
echo "To stop the application:"
echo "  podman pod rm -f app-pod"