"""
retriever.py
FAISS-based retrieval of similar past device logs.

How it works:
  1. Each time /triage processes a log, the resulting summary is embedded
     using sentence-transformers (all-MiniLM-L6-v2, 384-dim) and added
     to a FAISS flat L2 index.
  2. At inference time, the current log's anomaly text is embedded and
     the top-k nearest past logs are retrieved.
  3. Summaries of similar past cases are injected into the LLM prompt,
     giving the model concrete examples to reason against.

Index is persisted to disk in retriever_index/ so it survives server restarts.
If faiss-cpu or sentence-transformers are not installed, retrieval silently
degrades — the pipeline still works, just without the RAG context.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX_DIR     = Path("retriever_index")
EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output dimension
ENCODER_MODEL = "all-MiniLM-L6-v2"


class LogRetriever:
    def __init__(self):
        self._encoder   = None
        self._index     = None
        self._documents: list[dict] = []
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import faiss                          # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "faiss-cpu or sentence-transformers not installed. "
                "Retrieval disabled. "
                "Run: pip install faiss-cpu sentence-transformers"
            )
            return False

    def _ensure_loaded(self):
        """Lazy-load encoder and FAISS index on first use."""
        if self._encoder is not None:
            return

        import faiss
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(ENCODER_MODEL)

        index_file = INDEX_DIR / "index.faiss"
        docs_file  = INDEX_DIR / "documents.json"

        if index_file.exists():
            self._index     = faiss.read_index(str(index_file))
            self._documents = json.loads(docs_file.read_text()) if docs_file.exists() else []
            logger.info(f"Retrieval index loaded: {len(self._documents)} documents")
        else:
            self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
            logger.info("New retrieval index initialised (empty)")

    def add(self, summary: str, findings: dict):
        """
        Add a processed log to the retrieval index.
        Called after each successful /triage request.
        """
        if not self._available:
            return
        import numpy as np
        self._ensure_loaded()

        vec = self._encoder.encode([summary], normalize_embeddings=True)
        self._index.add(vec.astype("float32"))
        self._documents.append({
            "summary":  summary,
            "severity": findings.get("severity", "low"),
        })
        self._save()

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Return top-k similar past logs for a given query string.
        Returns [] if the index is empty or retrieval is unavailable.
        """
        if not self._available or not self._documents:
            return []
        import numpy as np
        self._ensure_loaded()

        vec = self._encoder.encode([query], normalize_embeddings=True)
        n   = min(k, len(self._documents))
        distances, indices = self._index.search(vec.astype("float32"), n)
        return [
            self._documents[i]
            for i in indices[0]
            if 0 <= i < len(self._documents)
        ]

    def _save(self):
        """Persist index and documents to disk."""
        import faiss
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self._index, str(INDEX_DIR / "index.faiss"))
        (INDEX_DIR / "documents.json").write_text(
            json.dumps(self._documents, indent=2)
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: LogRetriever | None = None


def get_retriever() -> LogRetriever:
    """Return the shared LogRetriever instance (created on first call)."""
    global _retriever
    if _retriever is None:
        _retriever = LogRetriever()
    return _retriever