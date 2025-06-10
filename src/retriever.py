import json
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer

from src.config import INDEX_PATH, META_PATH, EMBEDDING_MODEL


class LocalRetriever:
    """
    Returns top-k text chunks for a query from the FAISS index.
    Raises if index is missing, prompting ingestion first.
    """
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Vector index not found at {INDEX_PATH}.\n"
                "Please run ingestion: python -m src.ingest"
            )
        self.idx   = faiss.read_index(str(INDEX_PATH))
        self.meta  = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def __call__(self, query: str, k: int = 4) -> List[str]:
        qvec = self.model.encode(query, normalize_embeddings=True)
        D, I = self.idx.search(np.array([qvec], dtype="float32"), k)
        return [ self.meta[i]["text"] for i in I[0] ]