"""
Configuration settings for the RAG Wikipedia App.
Centralized so ingestion, vector store, and Streamlit app share the same config.
"""

import os

# === Paths ===
# Local directory where ChromaDB will persist its data
CHROMA_PERSIST_DIR = os.path.join("chromadb")

# === Embeddings ===
# Name of Ollama embedding model to use (example: "llama2" or "gemma:2b")
OLLAMA_EMBEDDING_MODEL = "gemma:2b"


# === LLM ===
# Name of Ollama LLM to use for answering questions
OLLAMA_LLM_MODEL = "llama3"

# === Ingestion ===
# Default Wikipedia language
WIKI_LANG = "en"

# === Streamlit ===
# Title for the app UI
APP_TITLE = "📚 RAG Wikipedia Assistant"

# === Wikipedia Query === #
# Wikipedia query to search for
WIKI_QUERY = 'PFC Levski Sofia'

# === Wikipedia max docs === #
WIKI_MAX_DOCS = 3