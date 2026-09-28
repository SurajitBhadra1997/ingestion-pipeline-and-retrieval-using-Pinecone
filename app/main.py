from fastapi import FastAPI, Depends

from app.dependencies import get_retrieval_service
from app.retrieval_service import RetrievalService
from app.schemas import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    IndexDocumentsRequest,
    ReindexRequest,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
)

app = FastAPI()


@app.post("/documents/index")
async def index_documents(
    request: IndexDocumentsRequest,
    service: RetrievalService = Depends(get_retrieval_service),
):

    await service.index_documents(request)

    return {"status": "Upserted", "documents_indexed": len(request.documents)}


@app.post(
    "/search",
    response_model=SearchResponse,
)
async def search_documents(
    request: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
):

    res = await service.search(request)

    return SearchResponse(results=res)


@app.post("/delete", response_model=DeleteDocumentResponse)
async def delete_document(
    request: DeleteDocumentRequest,
    service: RetrievalService = Depends(get_retrieval_service),
):

    await service.delete_document(request)

    return DeleteDocumentResponse(
        status= True,
        deleted_document_id=request.document_id
    )




