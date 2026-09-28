from typing import Protocol
import ollama

from app.config import get_settings

settings = get_settings()


class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


class OllamaEmbeddingProvider:

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        response = ollama.embed(model=settings.embedding_model_name, input=texts)

        return response["embeddings"]

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:

        response = ollama.embed(model=settings.embedding_model_name, input=query)

        return response["embeddings"][0]
