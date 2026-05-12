"""Upload routes — file upload with background ingestion pipeline."""

import os
import uuid
import logging
import traceback
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy import select, desc

from database import async_session, Document
from models import DocumentResponse, StatusResponse
from services.extractor import extract_text
from services.chunker import chunk_document
from services.embedder import embed_texts
from services.qdrant_service import insert_points

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])
docs_router = APIRouter(tags=["documents"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
ALLOWED_TYPES = {".pdf", ".docx"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

QDRANT_COLLECTION = "sss"  # hardcoded for now, will be dynamic in future


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------

async def _run_ingestion(doc_id: str, file_path: str, filename: str, file_type: str):
    """Full ingestion pipeline — runs as a FastAPI BackgroundTask.

    Steps:
        1. Extract text from uploaded file
        2. Chunk the text with metadata
        3. Embed all chunk texts in one batched OpenAI call
        4. Insert all chunks into Qdrant collection
        5. Update SQLite status to "ready" with chunk_count

    On any failure: status → "failed", log full traceback.
    """
    logger.info("🚀  [%s] Ingestion started — file=%s, type=%s", doc_id, filename, file_type)

    try:
        # Step 1 — Extract text
        logger.info("📄  [%s] Step 1/4 — Extracting text…", doc_id)
        text = extract_text(file_path, file_type)
        logger.info("📄  [%s] Extracted %d characters.", doc_id, len(text))

        # Step 2 — Chunk
        logger.info("✂️  [%s] Step 2/4 — Chunking document…", doc_id)
        chunks = chunk_document(text, doc_id, filename, file_type)
        logger.info("✂️  [%s] Created %d chunks.", doc_id, len(chunks))

        # Step 3 — Embed
        logger.info("🧠  [%s] Step 3/4 — Embedding %d chunks…", doc_id, len(chunks))
        chunk_texts = [c["text"] for c in chunks]
        vectors = embed_texts(chunk_texts)
        logger.info("🧠  [%s] Received %d embedding vectors.", doc_id, len(vectors))

        # Step 4 — Insert into Qdrant
        logger.info("📦  [%s] Step 4/4 — Inserting into Qdrant collection '%s'…", doc_id, QDRANT_COLLECTION)
        inserted = insert_points(QDRANT_COLLECTION, chunks, vectors)
        logger.info("📦  [%s] Inserted %d points into Qdrant.", doc_id, inserted)

        # Step 5 — Update SQLite → "ready"
        async with async_session() as session:
            result = await session.execute(
                select(Document).where(Document.doc_id == doc_id)
            )
            doc = result.scalar_one()
            doc.status = "ready"
            doc.chunk_count = len(chunks)
            await session.commit()

        logger.info("✅  [%s] Ingestion complete — status=ready, chunks=%d", doc_id, len(chunks))

    except Exception as exc:
        logger.error(
            "❌  [%s] Ingestion FAILED: %s\n%s",
            doc_id, exc, traceback.format_exc(),
        )
        # Update SQLite → "failed"
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(Document).where(Document.doc_id == doc_id)
                )
                doc = result.scalar_one()
                doc.status = "failed"
                doc.error_message = str(exc)
                await session.commit()
            logger.info("🔴  [%s] Status updated to 'failed' in database.", doc_id)
        except Exception as db_exc:
            logger.error("❌  [%s] Failed to update DB status: %s", doc_id, db_exc)


# ---------------------------------------------------------------------------
# POST /api/upload/local
# ---------------------------------------------------------------------------

@router.post("/local")
async def upload_local(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf and .docx are allowed.",
        )

    # Read file content
    content = await file.read()

    # Validate size
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds 20MB limit ({len(content) / (1024*1024):.1f}MB).",
        )

    # Ensure uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save file
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(content)

    file_type = ext.lstrip(".")
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    logger.info(
        "📁  File saved — doc_id=%s, filename=%s, type=%s, size=%d bytes",
        doc_id, file.filename, file_type, len(content),
    )

    # Insert row into SQLite with status "processing"
    async with async_session() as session:
        doc = Document(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            status="processing",
            uploaded_at=now,
        )
        session.add(doc)
        await session.commit()

    logger.info("🗄️  [%s] Database row created — status=processing", doc_id)

    # Kick off ingestion in background (non-blocking)
    background_tasks.add_task(_run_ingestion, doc_id, save_path, file.filename, file_type)
    logger.info("⏳  [%s] Background ingestion task queued.", doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "processing",
    }


# ---------------------------------------------------------------------------
# GET /api/status/{doc_id}
# ---------------------------------------------------------------------------

@docs_router.get("/status/{doc_id}", response_model=StatusResponse)
async def get_status(doc_id: str):
    async with async_session() as session:
        result = await session.execute(
            select(Document).where(Document.doc_id == doc_id)
        )
        doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return StatusResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        uploaded_at=doc.uploaded_at,
        error_message=doc.error_message,
    )


# ---------------------------------------------------------------------------
# GET /api/documents
# ---------------------------------------------------------------------------

@docs_router.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    async with async_session() as session:
        result = await session.execute(
            select(Document).order_by(desc(Document.uploaded_at))
        )
        docs = result.scalars().all()

    return [
        DocumentResponse(
            doc_id=d.doc_id,
            filename=d.filename,
            file_type=d.file_type,
            status=d.status,
            chunk_count=d.chunk_count,
            uploaded_at=d.uploaded_at,
        )
        for d in docs
    ]
