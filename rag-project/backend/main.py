import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db
from routes import router
from services.qdrant_service import ensure_collection

# ---------------------------------------------------------------------------
# Logging — ensure all pipeline logs appear in the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QDRANT_COLLECTION = "sss"  # hardcoded for now


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔧  Initializing database…")
    await init_db()
    logger.info("✅  Database ready.")

    logger.info("🔧  Ensuring Qdrant collection '%s' exists…", QDRANT_COLLECTION)
    ensure_collection(QDRANT_COLLECTION)
    logger.info("✅  Qdrant collection '%s' is ready.", QDRANT_COLLECTION)

    yield


app = FastAPI(title="IntelliDoc RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

