import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# from app.util import logger
from core.rag import RAGPipeline
from core.llm import LLM
from settings import OLLAMA_EMBEDDING_MODEL, WIKI_QUERY

if __name__ == '__main__':
    load_dotenv()

    LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
    print(f"Starting application '{LANGCHAIN_PROJECT}' ....")

    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    assert OPENAI_API_KEY
    print(f'OPENAI_API_KEY: {OPENAI_API_KEY[:3]}**{OPENAI_API_KEY[-3:]}')

    rag_pipeline = RAGPipeline.create_default(embedding_model_name=OLLAMA_EMBEDDING_MODEL,
                                              query=WIKI_QUERY)

    rag_pipeline.init()

    promt_template = """
    Answer this question using only the provided context:
    {question}

    Context:
    {context}
    """

    llm_model = OllamaLLM(model='gemma:2b')

    llm = LLM(promt_template, llm_model, rag_pipeline.create_retriever(), True)

    response = llm(input={
        'question': 'How many league titles Levski won?'
    },config={'configurable': {'session_id': 'sess1'}})
    print(response)