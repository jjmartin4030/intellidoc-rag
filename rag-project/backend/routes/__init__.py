from fastapi import APIRouter
from .upload import router as upload_router
from .upload import docs_router
from .chat import router as chat_router

router = APIRouter()
router.include_router(upload_router)
router.include_router(docs_router)
router.include_router(chat_router)
