import os
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph

from nodes import find_product_details, product_visible_features, product_description, product_title
from state import ProductDescriptionGeneratorState

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
assert OPENAI_API_KEY
print(f'----> OPENAI_API_KEY: {OPENAI_API_KEY[:3]}***{OPENAI_API_KEY[-3:]}')

LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
assert LANGCHAIN_API_KEY
print(f'----> LANGCHAIN_API_KEY: {LANGCHAIN_API_KEY[:3]}***{LANGCHAIN_API_KEY[-3:]}')

# Required by LangChain 
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
print(f'LANGCHAIN_PROJECT: {LANGCHAIN_PROJECT}')
print(f'----> LANGCHAIN_PROJECT: {LANGCHAIN_PROJECT}')

def generating_gate(state: ProductDescriptionGeneratorState):
    return state['product_presense_status']

def create_graph():
    graph_builder = StateGraph(ProductDescriptionGeneratorState)

    # Add nodes
    find_product_node = 'Search product'
    visible_feature_node = 'Generate product visble features'
    description_node = 'Generate product description'
    title_node = 'Generate product title'

    graph_builder.add_node(find_product_node, find_product_details)
    graph_builder.add_node(visible_feature_node, product_visible_features)
    graph_builder.add_node(description_node, product_description)
    graph_builder.add_node(title_node, product_title)

    # Add edges
    graph_builder.add_edge(START, find_product_node)
    graph_builder.add_conditional_edges(find_product_node, generating_gate, path_map={
        'NOT_FOUND': END,
        'FOUND': visible_feature_node
    })
    graph_builder.add_edge(visible_feature_node, description_node)
    graph_builder.add_edge(description_node, title_node)
    graph_builder.add_edge(title_node, END)

    graph = graph_builder.compile()
    return graph

product_description_graph = create_graph()
