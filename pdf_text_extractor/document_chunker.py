import logging
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        logging.info("✂️ Chunking documents")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = text_splitter.split_documents(documents)
        logging.info(f"📄 Split into {len(chunked_docs)} chunks")
        return chunked_docs