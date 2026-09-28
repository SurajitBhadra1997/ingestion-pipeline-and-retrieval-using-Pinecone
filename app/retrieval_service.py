from app.chunking import chunk_document
from app.embedding_provider import EmbeddingProvider
from app.schemas import (
    DeleteDocumentRequest,
    IndexDocumentsRequest,
    SearchRequest,
    SearchResponse,
)
from app.vector_store import VectorStore

from app.config import get_settings

settings = get_settings()


class RetrievalService:

    def __init__(self, embedding_vector=EmbeddingProvider, vector_store=VectorStore):
        self.embedding_provider = embedding_vector
        self.vector_store = vector_store

    async def index_documents(self, request: IndexDocumentsRequest) -> None:

        BATCH_SIZE = 5

        for size in range(0, len(request.documents), BATCH_SIZE):

            iteration_range = size + BATCH_SIZE

            batched_docs = request.documents[size:iteration_range]

            vectors = []

            batch_stale_ids = []

            for document in batched_docs:

                existing_chunks = await self.vector_store.list_document_chunk_ids(
                    document.document_id, request.namespace
                )
                chunks = chunk_document(
                    document.document_id,
                    document.text,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )

                text_list = [x.text for x in chunks]

                chunk_ids_list = [x.chunk_id for x in chunks]

                # print("chcunk_ids_lIST",chunk_ids_list)

                stale_chunk_ids = list(set(existing_chunks) - set(chunk_ids_list))

                print("diference", stale_chunk_ids)

                batch_stale_ids.extend(stale_chunk_ids)

                embedded_vectors = await self.embedding_provider.embed_documents(
                    text_list
                )

                for vector, chunk in zip(embedded_vectors, chunks):
                    vectors.append(
                        {
                            "id": chunk.chunk_id,
                            "metadata": {
                                "document_id": document.document_id,
                                "text": document.text,
                                "source": document.source,
                                "document_version": document.document_version,
                                "embedding_version": settings.embedding_version,
                                "chunking_version": settings.chunking_version,
                                "chunk_index": chunk.chunk_index,
                                "chunk text": chunk.text,
                            },
                            "values": vector,
                        }
                    )

            await self.vector_store.upsert(vectors, request.namespace)

            if batch_stale_ids:
                await self.vector_store.delete_ids(batch_stale_ids, request.namespace)

    async def search(self, request: SearchRequest) -> SearchResponse:
        emebedded_query = await self.embedding_provider.embed_query(request.query)
        metadata_filter = {}

        if request.document_id:
            metadata_filter["document_id"] = request.document_id

        top_chunks = await self.vector_store.search(
            emebedded_query,
            request.top_k,
            request.namespace,
            metadata_filter,
        )
        similarity_threshold = request.similarity_threshold

        search_result = []

        for chunk in top_chunks.matches:
            print(chunk)
            if chunk.score >= similarity_threshold:
                search_result.append(
                    {
                        "document_id": chunk.metadata["document_id"],
                        "text": chunk.metadata["text"],
                        "score": round(chunk.score, 2),
                        "chunk_id": chunk.id,
                        "chunk_index": chunk.metadata["chunk_index"],
                        "chunk_text": chunk.metadata["chunk text"],
                    }
                )

        return search_result

    async def delete_document(self, request: DeleteDocumentRequest):
        return await self.vector_store.delete(request.document_id, request.namespace)

   
