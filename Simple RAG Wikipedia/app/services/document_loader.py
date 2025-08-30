from typing import List
from langchain.document_loaders import WikipediaLoader
from langchain_core.documents import Document

from .base import BaseService

class DocumentLoader(BaseService):
    def __init__(self, query, lang='en', max_docs=5):
        self.query = query
        self.lang = lang
        self.max_docs = max_docs
        self.loader = WikipediaLoader(query=query,
                                      lang=lang,
                                      load_max_docs=max_docs)
    
    def load_documents(self) -> List[Document]:
        return self.loader.load()