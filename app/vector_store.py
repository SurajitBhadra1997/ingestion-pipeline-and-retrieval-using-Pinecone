from typing import Any, Dict, List
from pinecone import Pinecone

from app.config import get_settings

settings = get_settings()

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)


class VectorStore:

    async def upsert(self, vectors, namespace):
        return index.upsert(vectors=vectors, namespace=namespace)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        namespace: str,
        metadata_filter: dict[str, Any] | None = None,
    ):
        return index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter=metadata_filter,
        )

    async def delete(self, document_id: str, namespace: str):
        return index.delete(ids=[document_id], namespace=namespace)

    async def delete_ids(
        self,
        ids: list[str],
        namespace: str,
    ) -> None:

        return index.delete(ids=ids, namespace=namespace)

    async def list_document_chunk_ids(
        self,
        document_id: str,
        namespace: str,
    ) -> list[str]:
        prefix_to_match = f"{document_id}:chunk-"

        existing_chunks = []

        for ids in index.list(prefix=prefix_to_match, namespace=namespace):
            for id in ids.vectors:
                 existing_chunks.append(id.id)

        return existing_chunks
