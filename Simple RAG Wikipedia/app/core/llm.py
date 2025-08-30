from typing import Any, Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableConfig, RunnableWithMessageHistory
# In memory implementation of chat message history. Stores messages in a memory list.
from langchain_community.chat_message_histories import ChatMessageHistory
# Abstract base class for storing chat message history.
from langchain_core.chat_history import BaseChatMessageHistory

class LLM(Callable):
    def __init__(self, prompt_tmpl: str, llm: BaseChatModel, retriever: BaseRetriever,
                 with_history=True):
        self.llm = llm
        self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_messages(messages=[('human', prompt_tmpl)])

        rag_chain: Runnable =\
            {'context': retriever, 'question': RunnablePassthrough()} | self.prompt | self.llm
        
        if with_history:
            self.store = {}
            self.rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history=self.get_session_history)
        else:
            self.rag_chain = rag_chain

    def __call__(self, input: Any, config: RunnableConfig|None):
        return self.rag_chain.invoke(input, config)

    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]