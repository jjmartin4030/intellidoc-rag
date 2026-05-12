"""OpenAI Embeddings Service — batched text embedding using text-embedding-3-small."""

import logging
from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "text-embedding-3-small"  # 1536 dimensions


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single batched OpenAI API call.

    Args:
        texts: List of chunk texts to embed.

    Returns:
        List of embedding vectors (each 1536-dim float list), in the same
        order as the input texts.

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    if not texts:
        logger.warning("embed_texts called with empty list — returning []")
        return []

    logger.info(
        "📡  Calling OpenAI Embeddings API — model=%s, texts=%d",
        MODEL, len(texts),
    )

    try:
        response = _client.embeddings.create(input=texts, model=MODEL)
        vectors = [item.embedding for item in response.data]
        logger.info(
            "✅  Embeddings received — %d vectors, dim=%d",
            len(vectors), len(vectors[0]) if vectors else 0,
        )
        return vectors
    except Exception as exc:
        logger.error("❌  OpenAI Embeddings API failed: %s", exc)
        raise RuntimeError(f"OpenAI Embeddings API failed: {exc}") from exc
