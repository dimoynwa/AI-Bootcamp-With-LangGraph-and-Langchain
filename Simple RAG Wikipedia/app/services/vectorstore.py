from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .base import BaseService

class VectorstoreService(BaseService):
    """
    Wrapper service for interacting with a Chroma vector store.

    This class provides methods to store documents and perform similarity search
    queries using embeddings. It acts as a thin abstraction over the Chroma client.

    Attributes:
        chroma_db: An instance of the Chroma client connected to the specified host and port.
    """
    def __init__(self, host, port, embedding_fn):
        """
        Initialize the VectorstoreService.

        Args:
            host (str): The hostname or IP address of the Chroma server.
            port (int): The port number where the Chroma server is running.
            embedding_fn (Callable): A function or object that converts text into vector embeddings.
        """
        self.chroma_db = Chroma(host=host, port=port, embedding_function=embedding_fn)

    def store(self, docs: List[Document]) -> None:
        """
        Add a list of documents to the Chroma vector store.

        Args:
            docs (List[Document]): A list of LangChain Document objects to store.

        Returns:
            None
        """
        self.chroma_db.add_documents(docs)

    def similarity_search_with_score(self, query: str, **kwargs) -> list[tuple[Document, float]]:
        """
        Perform a similarity search and return documents along with similarity scores.

        Args:
            query (str): The query string to search for.
            **kwargs: Additional keyword arguments to pass to Chroma's similarity search.

        Returns:
            List[Tuple[Document, float]]: A list of tuples containing matching Document objects
                                          and their similarity scores.
        """
        self.chroma_db.similarity_search_with_score(query=query, kwargs=kwargs)

    def similarity_search(self, query: str, **kwargs) -> list[Document]:
        """
        Perform a similarity search and return matching documents.

        Args:
            query (str): The query string to search for.
            **kwargs: Additional keyword arguments to pass to Chroma's similarity search.

        Returns:
            List[Document]: A list of matching Document objects.
        """
        self.chroma_db.similarity_search(query=query, kwargs=kwargs)

    def as_retriever(self, **kwargs):
        return self.chroma_db.as_retriever(**kwargs)