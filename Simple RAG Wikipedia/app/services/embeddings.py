from typing import Callable, List
from langchain.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

class EmbeddingsService(Callable):
    def __init__(self, model_name):
        super().__init__()
        print(f'Create EmbeddingService with model_name: {model_name}')
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)

    def embed_texts(self, texts: list[str]) -> List[List[float]]:
        """Embed a batch of texts and return vectors."""
        return self.embeddings.embed_documents(texts)

    def embed_documents(self, documents: list[Document]) -> List[List[float]]:
        """Embed a batch of documents and return vectors."""
        return self.embeddings.embed_documents(documents)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(query)
    
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)