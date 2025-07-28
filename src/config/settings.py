import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-3-27b-it")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:5000/v1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))
# Shorter timeout to prevent UI hanging on slow responses
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CHAINLIT_HOST = os.getenv("CHAINLIT_HOST", "0.0.0.0")
CHAINLIT_PORT = int(os.getenv("CHAINLIT_PORT", "8000"))

# Vector database configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/home/models/FAISS_INGEST/vectorstore/db_faiss")

# Log configuration
logger.info(f"Connecting to vLLM server at: {INFERENCE_SERVER_URL}")
logger.info(f"Using model: {MODEL_ID}")
logger.info(f"Using vector database at: {VECTOR_DB_PATH}")

# Model configuration
# Maximum tokens in responses - must leave room for prompt tokens within context window
MAX_TOKENS = 512  # Reduced from 1024 to 512 for faster responses
TEMPERATURE = 0.2  # Reduced to 0.2 for more focused and deterministic responses

# Question rewrite settings
REWRITE_MODEL_ID = os.getenv("REWRITE_MODEL_ID", MODEL_ID)
# Enable/disable question rewriting step (set ENV ENABLE_Q_REWRITE=false to turn off)
ENABLE_Q_REWRITE = os.getenv("ENABLE_Q_REWRITE", "true").lower() in ("1", "true", "yes")

# Server verification endpoint (for diagnostics)
LLM_HEALTH_PATH = os.getenv("LLM_HEALTH_PATH", "/v1/models") 