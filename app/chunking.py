from dataclasses import dataclass
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int


def chunk_document(
    document_id: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_text(text)

    chunk_list = []

    for chunk_index, value in enumerate(chunks):
        chunk_list.append(
            DocumentChunk(
                chunk_id=f"{document_id}:chunk-{chunk_index}",
                document_id=document_id,
                text=value,
                chunk_index=chunk_index,
            )
        )

    return chunk_list
