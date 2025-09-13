import json
import os
from typing import List
from state import ProductDescriptionGeneratorState

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
mock_data_path = os.path.join(current_dir, "mock_data", "mock_data.json")


def find_product_details(
    state: ProductDescriptionGeneratorState,
) -> ProductDescriptionGeneratorState:
    assert os.path.exists(mock_data_path)

    with open(mock_data_path, "r") as data_file:
        products: List[ProductDescriptionGeneratorState] = json.load(data_file)

    assert len(products) > 0

    for product in products:
        if product["product_id"] == state["product_id"]:
            product["product_presense_status"] = "FOUND"
            return product
    return {"product_presense_status": "NOT_FOUND"}


def product_visible_features(
    state: ProductDescriptionGeneratorState,
) -> ProductDescriptionGeneratorState:
    # LLM temperature is a parameter that controls the randomness and creativity of a large language model's output by adjusting the probability distribution of next-word predictions.
    # Lower temperatures (e.g., <0.5) make the output more predictable and deterministic, favoring more common, "safer" words, which is good for factual accuracy.
    # Higher temperatures (e.g., >0.8) increase randomness, leading to more varied, creative, and surprising outputs, ideal for tasks like brainstorming or storytelling.
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    system_instructions = """
    You are a sales expert. Based on the provided image you should generate a precised description of the image for marketing purpose.

    REQUIREMENTS:
    - Maximum words: 100
    - Focus on physical appearance, visible features, and distinguishing elements.
    - Ensure the extracted information is factual and directly observable in the image.
    - Return information in bullet-points.
    """

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "url",
                    "url": state["product_image_url"],
                }
            ]
        ),
    ]
    image_attributes_response = llm.invoke(input=messages)

    return {"product_features_from_image": image_attributes_response.content}


def product_description(
    state: ProductDescriptionGeneratorState,
) -> ProductDescriptionGeneratorState:
    """Generate product description based on product features stored in the database"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

    detailed_message_prompt = """
    You are a marketting expert. You will receive a product details and you need to create a full detailed description.
    WHAT YOU RECEIVE:
    - product name
    - product features
    - producs specifications
    - visibile features

    REQUIREMENTS:
    - Maximum word lengths: 500
    - Do NOT focus too much on visible features
    - Tone should be formal
    """

    product_details = f"""
    Name: {state["product_name"]},
    Features: {state["product_features"]},
    Specifications: {state["product_specifications"]}
    Visibility features: {state["product_features_from_image"]}
    """

    input_messages = [
        SystemMessage(content=detailed_message_prompt),
        HumanMessage(content=product_details),
    ]

    description_response = llm.invoke(input=input_messages)
    state["product_description"] = description_response.content
    return state


def product_title(
    state: ProductDescriptionGeneratorState,
) -> ProductDescriptionGeneratorState:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

    title_prompt = f"""
    Based on product name and product detailed description generate a marketting title.

    Name: {state['product_name']}
    Description: {state['product_description']}

    REQUIREMENTS:
    - Up to 20 words
    """

    title_response = llm.invoke(input=[title_prompt])
    state['product_short_description'] = title_response.content
    return state
