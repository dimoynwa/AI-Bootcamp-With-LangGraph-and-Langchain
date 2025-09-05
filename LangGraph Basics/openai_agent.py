import os

from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]