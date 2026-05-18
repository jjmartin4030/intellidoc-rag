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
    SparseVectorParams,
    SparseVector,
    Modifier,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import QDRANT_URL, QDRANT_API_KEY
from services.embedder import embed_texts_dense, embed_texts_sparse

logger = logging.getLogger(__name__)

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)


def _get_collection_name(name: str) -> str:
    """Route operations for 'sss' to 'sss_v2' to avoid touching old data."""
    return "sss_v2" if name == "sss" else name


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
    actual_name = _get_collection_name(name)
    
    try:
        existing = [c.name for c in _client.get_collections().collections]
        if actual_name in existing:
            logger.info(
                "✅  Qdrant collection '%s' already exists — leaving it untouched.", actual_name
            )
            return

        logger.info(
            "🆕  Creating Qdrant collection '%s' (dense dim=%d, sparse IDF)…",
            actual_name, vector_size,
        )
        _client.create_collection(
            collection_name=actual_name,
            vectors_config={
                "dense": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF
                )
            }
        )
        logger.info("✅  Qdrant collection '%s' created successfully.", actual_name)
    except Exception as exc:
        logger.warning(
            "⚠️  Qdrant unreachable during startup — collection '%s' will be "
            "created lazily on first use.  Error: %s", actual_name, exc,
        )


def insert_points(
    collection_name: str,
    chunks: list[dict],
    vectors: list[dict],
) -> int:
    """Insert new points into a Qdrant collection.  NEVER updates existing points.

    Each point gets a fresh UUID4 as its id. Inserts in batches of 100.

    Args:
        collection_name: Target collection.
        chunks: List of chunk metadata dicts (from chunker.chunk_document).
        vectors: Corresponding embedding vectors (dense + sparse dict).

    Returns:
        Number of points inserted.

    Raises:
        ValueError: If chunks and vectors length mismatch.
        RuntimeError: If the Qdrant upsert call fails.
    """
    actual_name = _get_collection_name(collection_name)
    
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch"
        )

    points = []
    for chunk, vector_dict in zip(chunks, vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": vector_dict["dense"],
                "sparse": SparseVector(
                    indices=vector_dict["sparse"]["indices"],
                    values=vector_dict["sparse"]["values"],
                )
            },
            payload=chunk,  # full metadata dict from chunker
        )
        points.append(point)

    logger.info(
        "📤  Inserting %d points into Qdrant collection '%s' in batches of 100…",
        len(points), actual_name,
    )

    try:
        inserted = 0
        for i in range(0, len(points), 100):
            batch = points[i:i+100]
            _client.upsert(collection_name=actual_name, points=batch)
            inserted += len(batch)
            
        logger.info(
            "✅  Successfully inserted %d points into '%s'.",
            inserted, actual_name,
        )
        return inserted
    except Exception as exc:
        logger.error("❌  Qdrant insert_points failed: %s", exc)
        raise RuntimeError(f"Qdrant insert_points failed: {exc}") from exc


def hybrid_search(question: str, doc_id: str, top_k: int = 10) -> list[dict]:
    """Perform a hybrid search using dense and sparse vectors, fused with RRF.
    
    Args:
        question: User query.
        doc_id: Document ID to filter by.
        top_k: Number of chunks to return.
        
    Returns:
        List of chunks with RRF scores.
    """
    actual_name = _get_collection_name("sss")
    
    logger.info("🔍  Generating embeddings for hybrid search...")
    dense_vec = embed_texts_dense([question])[0]
    sparse_vec = embed_texts_sparse([question])[0]
    
    doc_filter = Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )
    
    logger.info("🔍  Executing Qdrant hybrid query (RRF) on '%s' (filter: doc_id=%s)...", actual_name, doc_id)
    try:
        query_response = _client.query_points(
            collection_name=actual_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=20,
                    filter=doc_filter,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    ),
                    using="sparse",
                    limit=20,
                    filter=doc_filter,
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        
        results = []
        for hit in query_response.points:
            results.append({
                "chunk_index": hit.payload.get("chunk_index", 0),
                "text": hit.payload.get("text", ""),
                "score": round(hit.score, 4),
                "doc_id": hit.payload.get("doc_id", doc_id),
                "filename": hit.payload.get("filename", "")
            })
            
        logger.info("✅  Hybrid search returned %d results. Top score: %.4f", len(results), results[0]["score"] if results else 0)
        return results
        
    except Exception as exc:
        logger.error("❌  Qdrant hybrid search failed: %s", exc)
        raise RuntimeError(f"Qdrant hybrid search failed: {exc}") from exc
