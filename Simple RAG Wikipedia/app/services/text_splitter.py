from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseService

class TextSplitterService(BaseService):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the text splitter.
        chunk_size: max size of each text chunk.
        chunk_overlap: overlap between consecutive chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of raw documents into smaller chunks.
        Returns a list of chunked Document objects.
        """
        splitted = self.splitter.split_documents(documents=documents)
        return splitted