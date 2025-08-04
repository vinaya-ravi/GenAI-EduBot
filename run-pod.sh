#!/usr/bin/env bash
set -euo pipefail

IMAGE="vllm-server:latest"

# 0) Clean up existing containers and pods (if they exist)
echo "Cleaning up existing containers and pods..."
podman container exists vllm && podman stop vllm && podman rm vllm || true
podman pod exists vllm-pod && podman pod stop vllm-pod && podman pod rm vllm-pod || true

# 1) build (only if missing or changed)
echo "Building container image..."
podman build -t "${IMAGE}" .

# 2) create a pod that maps port 5000
echo "Creating pod..."
podman pod create --name vllm-pod -p 5000:5000

# 3) run the container inside that pod
echo "Running container in pod..."
podman run \
  --pod vllm-pod \
  --device nvidia.com/gpu=0 \
  --device nvidia.com/gpu=1 \
  --device nvidia.com/gpu=2 \
  --device nvidia.com/gpu=3 \
  --security-opt=label=disable \
  --shm-size=1g \
  -d --name vllm \
  -v /home/haridoss/.cache/huggingface:/root/.cache/huggingface:Z \
  -e TORCH_COMPILE_MODE=reduce-overhead \
  -e TORCH_INDUCTOR_COMPILE_TO_EAGER=1 \
  -e TORCH_DYNAMO_DISABLE=1 \
  -e TORCH_CUDNN_V8_API_DISABLED=1 \
  "${IMAGE}"

echo "VLLM server is running in pod. Check status with 'podman ps' and 'podman pod ps'"
echo "If you're still having Python header issues, try this command to enter the container and troubleshoot:"
echo "podman exec -it vllm bash"
echo ""
echo "Inside the container, you can check if Python headers are installed correctly:"
echo "find /usr -name Python.h"
echo "python3 -c 'import sysconfig; print(sysconfig.get_path(\"include\"))'"
