from typing import Optional

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    document_id: str
    text: str
    source: str
    document_version: Optional[str]


class IndexDocumentsRequest(BaseModel):
    documents: list[DocumentInput]
    namespace: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=20)
    similarity_threshold: float
    namespace: str
    document_id: Optional[str] = None


class SearchResult(BaseModel):
    document_id: str
    text: str
    score: float
    chunk_id: str
    chunk_index: int
    chunk_text: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


class DeleteDocumentRequest(BaseModel):
    document_id: str
    namespace: str


class DeleteDocumentResponse(BaseModel):
    status: bool
    deleted_document_id: str
