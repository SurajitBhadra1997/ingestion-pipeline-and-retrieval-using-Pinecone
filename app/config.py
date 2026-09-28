from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model_name: str
    pinecone_api_key: str
    pinecone_index_name: str
    embedding_version: str
    chunking_version: str
    chunk_size: int
    chunk_overlap: int

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
