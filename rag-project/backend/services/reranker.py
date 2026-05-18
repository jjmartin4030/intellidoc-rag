"""Local cross-encoder reranking service."""

import logging
from typing import List, Dict, Any
import math

logger = logging.getLogger(__name__)

# Load cross-encoder at module level (lazy loaded or loaded on import)
try:
    from sentence_transformers import CrossEncoder
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info("⏳ Loading cross-encoder model '%s'...", MODEL_NAME)
    _reranker = CrossEncoder(MODEL_NAME)
    logger.info("✅ Cross-encoder model loaded successfully.")
except ImportError:
    logger.warning("⚠️ sentence-transformers not installed, reranker will not work.")
    _reranker = None
except Exception as exc:
    logger.error("❌ Failed to load cross-encoder model: %s", exc)
    _reranker = None


def _sigmoid(x: float) -> float:
    """Normalize raw logits to 0-1 range."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank_chunks(question: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Rerank chunks using a local cross-encoder.
    
    Args:
        question: User query.
        chunks: List of chunk dicts (from Qdrant search).
        top_n: Number of chunks to return.
        
    Returns:
        Top N chunks sorted by normalized rerank_score (0-1).
    """
    if not chunks:
        return []
        
    # If model failed to load, gracefully degrade by returning original chunks
    if _reranker is None:
        logger.warning("⚠️ Reranker model unavailable. Falling back to original order.")
        for chunk in chunks:
            chunk["rerank_score"] = 0.0
        return chunks[:top_n]
        
    logger.info("⚖️ Reranking %d chunks...", len(chunks))
    
    try:
        # Build pairs: (question, chunk_text)
        pairs = [[question, chunk.get("text", "")] for chunk in chunks]
        
        # Predict logits
        logits = _reranker.predict(pairs)
        
        # Normalize and attach to chunks
        for i, chunk in enumerate(chunks):
            # predict() returns an array or single float; handle both
            val = float(logits[i]) if hasattr(logits, "__len__") else float(logits)
            chunk["rerank_score"] = _sigmoid(val)
            
        # Sort descending by rerank_score
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info("✅ Reranking complete. Top score: %.4f", reranked[0]["rerank_score"] if reranked else 0.0)
        return reranked[:top_n]
        
    except Exception as exc:
        logger.error("❌ Reranking failed: %s", exc)
        # Graceful fallback
        for chunk in chunks:
            chunk.setdefault("rerank_score", 0.0)
        return chunks[:top_n]
