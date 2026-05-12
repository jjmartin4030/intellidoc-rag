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
    status: str
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}
