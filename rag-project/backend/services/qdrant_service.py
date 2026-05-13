"""Qdrant Service — safe collection management and point insertion.

RULES (non-negotiable):
  • NEVER delete, recreate, modify, or update any existing collection or vector.
  • On startup: create collection ONLY if it does not already exist.
  • ONLY insert new points with fresh UUIDs.
"""

import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from config import QDRANT_URL, QDRANT_API_KEY

logger = logging.getLogger(__name__)

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)


def ensure_collection(name: str, vector_size: int = 1536) -> None:
    """Create a Qdrant collection ONLY if it does not already exist.

    This function will NEVER delete, recreate, or modify an existing
    collection.  It is safe to call on every startup.

    If Qdrant is unreachable the app will still start — the collection
    will be lazily created on the first ingestion request.

    Args:
        name: Collection name.
        vector_size: Embedding dimension (default 1536 for text-embedding-3-small).
    """
    try:
        existing = [c.name for c in _client.get_collections().collections]
        if name in existing:
            logger.info(
                "✅  Qdrant collection '%s' already exists — leaving it untouched.", name
            )
            return

        logger.info(
            "🆕  Creating Qdrant collection '%s' (dim=%d, distance=Cosine)…",
            name, vector_size,
        )
        _client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("✅  Qdrant collection '%s' created successfully.", name)
    except Exception as exc:
        logger.warning(
            "⚠️  Qdrant unreachable during startup — collection '%s' will be "
            "created lazily on first use.  Error: %s", name, exc,
        )


def insert_points(
    collection_name: str,
    chunks: list[dict],
    vectors: list[list[float]],
) -> int:
    """Insert new points into a Qdrant collection.  NEVER updates existing points.

    Each point gets a fresh UUID4 as its id.

    Args:
        collection_name: Target collection.
        chunks: List of chunk metadata dicts (from chunker.chunk_document).
        vectors: Corresponding embedding vectors (same order / length).

    Returns:
        Number of points inserted.

    Raises:
        ValueError: If chunks and vectors length mismatch.
        RuntimeError: If the Qdrant upsert call fails.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
        )

    points = []
    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=chunk,  # full metadata dict from chunker
        )
        points.append(point)

    logger.info(
        "📤  Inserting %d points into Qdrant collection '%s'…",
        len(points), collection_name,
    )

    try:
        _client.upsert(collection_name=collection_name, points=points)
        logger.info(
            "✅  Successfully inserted %d points into '%s'.",
            len(points), collection_name,
        )
        return len(points)
    except Exception as exc:
        logger.error("❌  Qdrant insert_points failed: %s", exc)
        raise RuntimeError(f"Qdrant insert_points failed: {exc}") from exc
