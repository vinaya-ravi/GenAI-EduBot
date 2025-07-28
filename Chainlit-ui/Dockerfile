FROM python:3.10-slim

# ---------- system ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ---------- user ----------
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /app
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONPATH=/app

# ---------- python ----------
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- code ----------
COPY --chown=appuser:appuser src/ ./src/

# ---------- runtime ----------
ENV INFERENCE_SERVER_URL=http://host.containers.internal:5000/v1 \
    VECTOR_DB_PATH=/home/models/FAISS_INGEST/vectorstore/db_faiss \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000

EXPOSE 8000
CMD ["chainlit","run","src/app.py","--host","0.0.0.0","--port","8000"]
