from services.document_loader import DocumentLoader
from services.text_splitter import TextSplitterService
from services.embeddings import EmbeddingsService
from services.vectorstore import VectorstoreService

from config import CHROMA_HOST, CHROMA_PORT, OLLAMA_EMBEDDING_MODEL, WIKI_LANG, WIKI_MAX_DOCS, WIKI_QUERY

from langchain_core.retrievers import BaseRetriever

class RAGPipeline:
    """Main RAG pipeline that orchestrates all services"""
    def __init__(
        self,
        document_loader: DocumentLoader,
        text_splitter: TextSplitterService,
        embeddings: EmbeddingsService,
        vectorstore: VectorstoreService
    ):
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    @classmethod
    def create_default(cls, 
                       embedding_model_name=OLLAMA_EMBEDDING_MODEL,
                       query=WIKI_QUERY,
                       lang=WIKI_LANG,
                       max_docs=WIKI_MAX_DOCS,
                       chunk_size=1000,
                       chunk_overlap=50) -> 'RAGPipeline':
        """Factory method to create a pipeline with default dependencies"""
        
        # Create services
        document_loader = DocumentLoader(query=query, lang=lang, max_docs=max_docs)
        text_splitter = TextSplitterService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = EmbeddingsService(model_name=embedding_model_name)
        vectorstore = VectorstoreService(host=CHROMA_HOST,
                                         port=CHROMA_PORT,
                                         embedding_fn=embeddings)
        
        return cls(document_loader, text_splitter, embeddings, vectorstore)
    
    def init(self):
        print('Fetching documents from Wikipedia...')
        documents = self.document_loader.load_documents()
        print(f'Fetched {len(documents)} from Wikipedia')
        print('Spliting documents...')
        split_docs = self.text_splitter.split(documents=documents)
        print(f'Documents split into {len(split_docs)} chunks')
        print('Storing into Vector DB...')
        self.vectorstore.store(split_docs)
        print('Documents stored.')

    def create_retriever(self, **kwargs) -> BaseRetriever:
        return self.vectorstore.as_retriever(**kwargs)