"""OpenAI Embeddings Service — batched text embedding using text-embedding-3-small."""

import logging
from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "text-embedding-3-small"  # 1536 dimensions

# Load sparse model at module level
try:
    from fastembed import SparseTextEmbedding
    logger.info("⏳ Loading sparse embedding model 'prithivida/Splade_PP_EN_V1'...")
    _sparse_model = SparseTextEmbedding("prithivida/Splade_PP_EN_V1")
    logger.info("✅ Sparse embedding model loaded successfully.")
except ImportError:
    logger.warning("⚠️ fastembed not installed, sparse embeddings will fail.")
    _sparse_model = None
except Exception as exc:
    logger.error("❌ Failed to load sparse embedding model: %s", exc)
    _sparse_model = None


def embed_texts_dense(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using OpenAI dense vectors."""
    if not texts:
        logger.warning("embed_texts_dense called with empty list — returning []")
        return []

    logger.info("📡  Calling OpenAI Embeddings API — model=%s, texts=%d", MODEL, len(texts))
    try:
        response = _client.embeddings.create(input=texts, model=MODEL)
        vectors = [item.embedding for item in response.data]
        logger.info("✅  Dense embeddings received — %d vectors, dim=%d", len(vectors), len(vectors[0]) if vectors else 0)
        return vectors
    except Exception as exc:
        logger.error("❌  OpenAI Embeddings API failed: %s", exc)
        raise RuntimeError(f"OpenAI Embeddings API failed: {exc}") from exc


def embed_texts_sparse(texts: list[str]) -> list[dict]:
    """Embed a list of texts using local fastembed sparse model."""
    if not texts:
        return []
        
    if _sparse_model is None:
        raise RuntimeError("Sparse embedding model is not loaded.")
        
    logger.info("📡  Calling Sparse Embedder — texts=%d", len(texts))
    try:
        # returns list of SparseEmbedding objects with .indices and .values
        results = list(_sparse_model.embed(texts))
        
        sparse_vectors = []
        for res in results:
            # fastembed might return numpy arrays, convert to lists
            sparse_vectors.append({
                "indices": res.indices.tolist() if hasattr(res.indices, "tolist") else list(res.indices),
                "values": res.values.tolist() if hasattr(res.values, "tolist") else list(res.values)
            })
            
        logger.info("✅  Sparse embeddings received — %d vectors", len(sparse_vectors))
        return sparse_vectors
    except Exception as exc:
        logger.error("❌  Sparse Embeddings API failed: %s", exc)
        raise RuntimeError(f"Sparse Embeddings API failed: {exc}") from exc


def embed_texts(texts: list[str]) -> list[dict]:
    """Generate both dense and sparse embeddings.
    
    Returns:
        List of dicts: [{"dense": [...], "sparse": {"indices": [...], "values": [...]}}, ...]
    """
    dense_vecs = embed_texts_dense(texts)
    sparse_vecs = embed_texts_sparse(texts)
    
    if len(dense_vecs) != len(sparse_vecs):
        raise ValueError("Mismatch between dense and sparse vector counts.")
        
    combined = []
    for d, s in zip(dense_vecs, sparse_vecs):
        combined.append({
            "dense": d,
            "sparse": s
        })
        
    return combined
