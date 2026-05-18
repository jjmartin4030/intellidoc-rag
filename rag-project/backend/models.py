from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: Optional[int] = None
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class StatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    uploaded_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    doc_id: str
    question: str


class ChunkResult(BaseModel):
    chunk_index: int
    text: str
    score: float
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    doc_id: str
    filename: str
    is_out_of_context: bool
    source_chunks: list[ChunkResult]
    top_score: float
    top_rerank_score: float
