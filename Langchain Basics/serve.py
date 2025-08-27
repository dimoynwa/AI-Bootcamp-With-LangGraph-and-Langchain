import os
from dotenv import load_dotenv

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq

from langserve import add_routes

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']
assert GROQ_API_KEY

model = ChatGroq(model='Gemma2-9b-It', api_key=GROQ_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(messages=[
    ('system', 'Translate this from English to Bulgarian in most informal way. Try to use some swag. Just answer, without explanation'),
    ('user', '{input}')
])

output_parser = StrOutputParser()

chain = prompt_template | model | output_parser

## App definition
app = FastAPI(title='Langchain server', version='1.0.0',
              description='A simple API server using Langchain and Langserve')
add_routes(app=app,
           runnable=chain,
           path='/api/chain')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)