"""Chat route — question answering over uploaded documents via RAG."""

import logging
from openai import OpenAI
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from qdrant_client.http.models import FieldCondition, MatchValue, Filter

from config import OPENAI_API_KEY
from database import async_session, Document
from models import ChatRequest, ChatResponse, ChunkResult
from services.embedder import embed_texts
from services.qdrant_service import _client as qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

QDRANT_COLLECTION = "sss"
SCORE_THRESHOLD = 0.40
TOP_K = 5

_openai = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY "
    "the context provided below extracted from the uploaded document.\n"
    "Do not use any outside knowledge whatsoever.\n"
    "Be concise and factual.\n"
    "If the answer cannot be found in the context, respond with exactly:\n"
    "'This question is outside the scope of the uploaded document.'"
)

OUT_OF_CONTEXT_ANSWER = "This question is outside the scope of the uploaded document."
OUT_OF_CONTEXT_PHRASE = "outside the scope of the uploaded document"


# ---------------------------------------------------------------------------
# POST /api/chat/
# ---------------------------------------------------------------------------

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    doc_id = req.doc_id.strip()
    question = req.question.strip()

    # ── Validation ────────────────────────────────────────────────────────
    if not doc_id:
        logger.warning("⛔  Chat request rejected — empty doc_id")
        raise HTTPException(status_code=400, detail="doc_id is required.")

    if not question:
        logger.warning("⛔  Chat request rejected — empty question (doc_id=%s)", doc_id)
        raise HTTPException(status_code=400, detail="question is required.")

    if len(question) > 1000:
        logger.warning(
            "⛔  Chat request rejected — question too long (%d chars, doc_id=%s)",
            len(question), doc_id,
        )
        raise HTTPException(
            status_code=400,
            detail="question must not exceed 1000 characters.",
        )

    logger.info("💬  Chat request — doc_id=%s, question=%d chars", doc_id, len(question))

    # ── Lookup document in SQLite ─────────────────────────────────────────
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()

    if doc is None:
        logger.warning("⛔  Document not found — doc_id=%s", doc_id)
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    if doc.status != "ready":
        logger.warning(
            "⛔  Document not ready — doc_id=%s, status=%s", doc_id, doc.status
        )
        raise HTTPException(
            status_code=400,
            detail="Document is still processing, please wait.",
        )

    filename = doc.filename
    logger.info("📄  Document found — doc_id=%s, filename=%s, status=%s", doc_id, filename, doc.status)

    # ── Step 1: Embed the question ────────────────────────────────────────
    logger.info("🧠  Step 1/4 — Embedding question…")
    try:
        question_vector = embed_texts([question])[0]
        logger.info("🧠  Question embedded — dim=%d", len(question_vector))
    except Exception as exc:
        logger.error("❌  Failed to embed question: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to embed question. Please try again later.",
        )

    # ── Step 2: Search Qdrant ─────────────────────────────────────────────
    logger.info(
        "🔍  Step 2/4 — Searching Qdrant collection '%s' (top_k=%d, filter=doc_id:%s)…",
        QDRANT_COLLECTION, TOP_K, doc_id,
    )
    try:
        query_response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=question_vector,
            limit=TOP_K,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        search_results = query_response.points
        logger.info("🔍  Qdrant returned %d results.", len(search_results))
    except Exception as exc:
        logger.error("❌  Qdrant search failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Vector search failed. Please try again later.",
        )

    # ── Step 3: Score check ───────────────────────────────────────────────
    if not search_results:
        logger.info("⚠️  No chunks retrieved — returning out-of-context response.")
        return ChatResponse(
            answer=OUT_OF_CONTEXT_ANSWER,
            doc_id=doc_id,
            filename=filename,
            is_out_of_context=True,
            source_chunks=[],
            top_score=0.0,
        )

    top_score = search_results[0].score
    logger.info("📊  Top score: %.4f (threshold: %.2f)", top_score, SCORE_THRESHOLD)

    # Build source_chunks for the response (sorted by chunk_index)
    source_chunks = sorted(
        [
            ChunkResult(
                chunk_index=hit.payload.get("chunk_index", 0),
                text=hit.payload.get("text", ""),
                score=round(hit.score, 4),
            )
            for hit in search_results
        ],
        key=lambda c: c.chunk_index,
    )

    if top_score < SCORE_THRESHOLD:
        logger.info(
            "⚠️  Top score %.4f < threshold %.2f — skipping LLM, returning out-of-context.",
            top_score, SCORE_THRESHOLD,
        )
        return ChatResponse(
            answer=OUT_OF_CONTEXT_ANSWER,
            doc_id=doc_id,
            filename=filename,
            is_out_of_context=True,
            source_chunks=source_chunks,
            top_score=round(top_score, 4),
        )

    # ── Step 4: Build context and call GPT-4o ─────────────────────────────
    logger.info("📝  Step 3/4 — Building context from %d chunks…", len(source_chunks))
    context_blocks = "\n".join(
        f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(source_chunks)
    )

    user_message = f"Context:\n{context_blocks}\n\nQuestion: {question}"

    logger.info("🤖  Step 4/4 — Calling GPT-4o (temperature=0, max_tokens=1000)…")
    try:
        completion = _openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer = completion.choices[0].message.content.strip()
        logger.info(
            "✅  GPT-4o responded — %d chars, tokens(prompt=%d, completion=%d)",
            len(answer),
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
    except Exception as exc:
        logger.error("❌  GPT-4o call failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="LLM generation failed. Please try again later.",
        )

    # ── Detect out-of-context from LLM response ──────────────────────────
    is_out_of_context = OUT_OF_CONTEXT_PHRASE in answer.lower()
    if is_out_of_context:
        logger.info("⚠️  LLM flagged answer as out-of-context.")

    logger.info(
        "✅  Chat complete — doc_id=%s, is_out_of_context=%s, top_score=%.4f",
        doc_id, is_out_of_context, top_score,
    )

    return ChatResponse(
        answer=answer,
        doc_id=doc_id,
        filename=filename,
        is_out_of_context=is_out_of_context,
        source_chunks=source_chunks,
        top_score=round(top_score, 4),
    )
