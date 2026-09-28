from app.embedding_provider import OllamaEmbeddingProvider
from app.vector_store import VectorStore
from app.retrieval_service import RetrievalService

embedding_provider = OllamaEmbeddingProvider()


vector_store = VectorStore()


def get_retrieval_service() -> RetrievalService:

    return RetrievalService(embedding_provider,vector_store)
