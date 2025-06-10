import os
from pathlib import Path

# OpenAI and embedding configuration
API_KEY         = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Chunk parameters for token splitting
CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 100

# Data directories
BASE_DIR        = Path(__file__).parent.parent
INCOMING_DIR    = BASE_DIR / "data" / "incoming"
ARCHIVE_DIR     = BASE_DIR / "data" / "archive"
VECTOR_DIR      = BASE_DIR / "data" / "vector_store"

# Vector store files
INDEX_PATH      = VECTOR_DIR / "index.faiss"
META_PATH       = VECTOR_DIR / "metadata.jsonl"