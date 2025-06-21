#!/usr/bin/env python3
"""
server.py: Flask-based API for PDF RAG System
Combines pipeline functions and REST endpoints into a single module.
"""
import os
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from huggingface_hub import HfApi
# --- Pipeline Functions ---
from getpass import getpass

def setup_huggingface_token(token: str = None, token_only: bool = False):
    """
    Setup HuggingFace authentication token.
    Prioritizes: 1. token argument, 2. HUGGINGFACEHUB_API_TOKEN env var, 3. getpass prompt.
    If token_only, skip print and return user info.
    """
    final_token = None

    if token:
        # 1. Use the token passed as an argument
        final_token = token
    else:
        # 2. Try to get the token from the environment variable
        env_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if env_token:
            final_token = env_token
        else:
            # 3. If no token found anywhere, prompt if not in token_only mode
            #    or if this is meant to be an interactive setup.
            #    For API calls, 'token' should usually be provided or already in env.
            if not token_only: # Only prompt if not in a "token-only" context (which implies non-interactive)
                               # and if no token was found.
                print("Hugging Face token not found in environment. Please provide it.")
                final_token = getpass("Paste your Hugging Face token here: ")
            else:
                # If token_only is True, and no token provided, and no token in env, it's an error
                raise ValueError("Hugging Face token is required but not provided or found in environment.")

    if not final_token:
        raise ValueError("Hugging Face token is required but could not be obtained.")

    # Ensure the token is set in the environment for consistency
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = final_token

    api = HfApi()
    try:
        user = api.whoami(token=final_token) # Use the obtained final_token for authentication
    except Exception as e:
        # Re-raise with a more specific message for easier debugging
        raise Exception(f"Failed to authenticate with Hugging Face token: {e}")

    if token_only:
        return user
    print(f"Authenticated as: {user.get('name', user.get('user'))}")
    return user

def explore_dataset(dataset_path: str = "./"):
    """List files in the specified directory (no recursion)."""
    files = []
    for f in os.listdir(dataset_path):
        if os.path.isfile(os.path.join(dataset_path, f)):
            files.append(f)
    return files


def load_pdf_document(pdf_path: str):
    """Load PDF pages as LangChain documents."""
    from langchain_community.document_loaders.pdf import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs


def split_text_into_chunks(docs):
    """Split loaded documents into overlapping text chunks."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    return chunks


def create_vector_store(chunks, persist_dir: str = "./chroma_db"):
    """Embed chunks and persist a Chroma vector store."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain.vectorstores import Chroma
    model_name = "thenlper/gte-large"
    embedding_model = FastEmbedEmbeddings(model_name=model_name)
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_dir)
    return db


def setup_retriever(db, k: int = 4):
    """Configure retriever with Maximum Marginal Relevance."""
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    return retriever


def setup_llm(model: str = "HuggingFaceH4/zephyr-7b-beta"):
    """Instantiate a HuggingFaceEndpoint LLM for text generation."""
    from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        model=model,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )
    return llm


def create_rag_chain(retriever, llm):
    """Assemble a retrieval-augmented generation chain."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    template = (
        "<s>[INST]\n"
        "You are an AI Assistant that follows instructions extremely well."
        "Be truthful and give direct answers, or 'I don't know' if out of context.\n"
        "[/INST]\n"
        "CONTEXT: {context}\n"
        "</s>\n"
        "[INST]\n"
        "{query}\n"
        "[/INST]"
    )
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return chain


def ask_question(chain, question: str):
    """Invoke the RAG chain on a user query."""
    return chain.invoke(question)
