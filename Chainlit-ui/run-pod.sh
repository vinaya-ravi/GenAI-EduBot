#!/usr/bin/env bash
set -euo pipefail

IMAGE="chainlit-ui:latest"

podman build -t "${IMAGE}" .

# own network namespace, port 8000
podman pod create --name chainlit-pod -p 8000:8000 || true

podman run \
  --pod chainlit-pod \
  --security-opt=label=disable \
  -d --name chainlit \
  -v /home/haridoss/gemma/models/FAISS_INGEST:/home/models/FAISS_INGEST:Z \
  "${IMAGE}"
