import json
import shutil
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    INCOMING_DIR, ARCHIVE_DIR,
    VECTOR_DIR, INDEX_PATH, META_PATH
)


def split_to_chunks(text: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int    = CHUNK_OVERLAP):
    """
    Token-based splitter using tiktoken to ensure chunks fit GPT-4.1's context.
    """
    enc = tiktoken.encoding_for_model("gpt-4o-full")
    tokens = enc.encode(text)
    step   = chunk_size - overlap
    for start in range(0, len(tokens), step):
        yield enc.decode(tokens[start:start + chunk_size])


def ingest():
    """
    Ingest pipeline: split → embed → store → archive
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Load or initialize FAISS index & metadata
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        meta  = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]
    else:
        dim   = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        meta  = []

    # Process each incoming file
    for path in INCOMING_DIR.glob("*.*"):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_to_chunks(text)):
            vec = model.encode(chunk, normalize_embeddings=True)
            index.add(np.array([vec], dtype="float32"))
            meta.append({
                "id":    len(meta),
                "file":  path.name,
                "chunk": i,
                "text":  chunk
            })
        shutil.move(str(path), ARCHIVE_DIR / path.name)

    # Save index & metadata
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        for record in meta:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ingest()