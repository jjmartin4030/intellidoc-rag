"""Chat route — question answering over uploaded documents via RAG."""

import logging
from openai import OpenAI
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from config import OPENAI_API_KEY
from database import async_session, Document
from models import ChatRequest, ChatResponse, ChunkResult
from services.qdrant_service import hybrid_search
from services.reranker import rerank_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

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


def is_out_of_context(chunks: list[dict]) -> bool:
    """Dynamic thresholding based on cross-encoder rerank scores."""
    if not chunks:
        return True
        
    top_score = chunks[0].get("rerank_score", 0.0)
    
    if top_score < 0.20:
        return True
    if top_score >= 0.40:
        return False
        
    if len(chunks) > 1:
        gap = top_score - chunks[1].get("rerank_score", 0.0)
        if gap > 0.05:
            return False
            
    return True


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

    # ── Step 1: Hybrid Search ─────────────────────────────────────────────
    logger.info("🔍  Step 1/4 — Hybrid Search on Qdrant (doc_id=%s)…", doc_id)
    try:
        search_results = hybrid_search(question, doc_id, top_k=10)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Hybrid search failed.")
        
    if not search_results:
        logger.info("⚠️  No chunks retrieved — returning out-of-context response.")
        return ChatResponse(
            answer=OUT_OF_CONTEXT_ANSWER,
            doc_id=doc_id,
            filename=filename,
            is_out_of_context=True,
            source_chunks=[],
            top_score=0.0,
            top_rerank_score=0.0,
        )
        
    top_rrf_score = search_results[0].get("score", 0.0)

    # ── Step 2: Cross-encoder Reranking ───────────────────────────────────
    logger.info("⚖️  Step 2/4 — Reranking %d chunks…", len(search_results))
    reranked_chunks = rerank_chunks(question, search_results, top_n=5)
    
    top_rerank_score = reranked_chunks[0].get("rerank_score", 0.0)
    
    # Sort models by chunk_index for LLM context, but keeping rerank order is also fine.
    # We will build context models sorted by chunk_index ascending for logical reading.
    source_chunks_models = sorted([
        ChunkResult(
            chunk_index=c.get("chunk_index", 0),
            text=c.get("text", ""),
            score=c.get("score", 0.0),
            rerank_score=c.get("rerank_score", 0.0),
        )
        for c in reranked_chunks
    ], key=lambda x: x.chunk_index)

    # ── Step 3: Dynamic Threshold Check ───────────────────────────────────
    logger.info("📊  Step 3/4 — Evaluating dynamic threshold (top_score=%.4f)…", top_rerank_score)
    out_of_context = is_out_of_context(reranked_chunks)
    
    if out_of_context:
        logger.info("⚠️  Rerank scores indicate out of context — skipping LLM.")
        return ChatResponse(
            answer=OUT_OF_CONTEXT_ANSWER,
            doc_id=doc_id,
            filename=filename,
            is_out_of_context=True,
            source_chunks=source_chunks_models,
            top_score=top_rrf_score,
            top_rerank_score=top_rerank_score,
        )

    # ── Step 4: Build context and call GPT-4o ─────────────────────────────
    logger.info("📝  Step 4/4 — Building context and calling GPT-4o…")
    context_blocks = "\n".join(
        f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(source_chunks_models)
    )

    user_message = f"Context:\n{context_blocks}\n\nQuestion: {question}"

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
    llm_out_of_context = OUT_OF_CONTEXT_PHRASE in answer.lower()
    if llm_out_of_context:
        logger.info("⚠️  LLM flagged answer as out-of-context.")

    logger.info(
        "✅  Chat complete — doc_id=%s, is_out_of_context=%s",
        doc_id, llm_out_of_context,
    )

    return ChatResponse(
        answer=answer,
        doc_id=doc_id,
        filename=filename,
        is_out_of_context=llm_out_of_context,
        source_chunks=source_chunks_models,
        top_score=round(top_rrf_score, 4),
        top_rerank_score=round(top_rerank_score, 4),
    )
