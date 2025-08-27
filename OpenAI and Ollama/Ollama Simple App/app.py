import os
from dotenv import load_dotenv

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

# Load the environment variables in .env file
load_dotenv()

# We will use LANGCHAIN_API_KEY for LangSmith tracking
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
if not LANGCHAIN_API_KEY:
    print('LANGCHAIN_API_KEY is required. Please provide one...')
    exit(1)
print(f'LANGCHAIN_API_KEY: {LANGCHAIN_API_KEY[:10]}***{LANGCHAIN_API_KEY[-3:]}')

# Required by LangChain 
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
print(f'LANGCHAIN_PROJECT: {LANGCHAIN_PROJECT}')


## Prompt Template 
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a sarcastic assistant, please respond to the question asked using some dad joke'),
    ('user', 'Question: {question}')
])

## Streamlit framework
st.title('Langchain demo with Google Gemma 2b model')
input_text = st.text_input('Ask me anything...')

## Ollama Gemma:2b model
llm = Ollama(model='gemma:2b')
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

if input_text:
    llm_response = chain.invoke(input={'question': input_text})
    st.write(llm_response)