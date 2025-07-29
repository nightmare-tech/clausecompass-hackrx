import logging
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, HttpUrl

# --- Core AI/ML & Data Libraries ---
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
# --- NEW IMPORTS for Google Gemini ---
from google import genai
# --- END NEW IMPORTS ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

executor = ThreadPoolExecutor(max_workers=10)

# --- Initialize Global Objects ---
app = FastAPI(
    title="HackRx 6.0 RAG API - ClauseCompasies",
    description="An LLM-Powered Intelligent Query–Retrieval System for Bajaj Finserv Health."
)

# --- NEW: Load Google Gemini Client ---
client = genai.Client()
# --- END NEW CLIENT ---

# Load Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    logger.info(f"Successfully loaded LangChain-wrapped embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}", exc_info=True)
    raise

# Hardcoded Bearer Token for the hackathon
EXPECTED_BEARER_TOKEN = "e66c2e8eb6884ded2c7177421784e760b34b9297bfebc20a2a272cc63357270d"

# --- Pydantic Models for API Schema ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Authentication Dependency ---
async def verify_token(Authorization: str = Header(..., description="Bearer token for authentication.")):
    if Authorization != f"Bearer {EXPECTED_BEARER_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authentication token.")

# --- Core RAG Logic Functions ---
def process_document_from_url(doc_url: str) -> FAISS:
    # This function does not need to change. It's already perfect.
    tmp_file_path = None
    try:
        logger.info(f"Downloading document from URL: {doc_url}")
        response = requests.get(doc_url, stream=True)
        response.raise_for_status()
        file_extension = Path(doc_url.split('?')[0]).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        logger.info(f"Document saved to temporary file: {tmp_file_path}")

        loader = UnstructuredFileLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract content from the provided document.")
        
        logger.info(f"Document chunked into {len(chunks)} pieces. Creating FAISS index...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        logger.info("In-memory FAISS index created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the document.")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def answer_question_with_rag(question: str, vector_store: FAISS) -> str:
    """Performs a RAG query to answer a single question using Google Gemini."""
    try:
        logger.info(f"Performing similarity search for question: '{question}'")
        # Be more generous with retrieval, since Gemini's context window is huge
        retrieved_docs = vector_store.similarity_search(question, k=10) # Increased k to 10
        
        if not retrieved_docs:
            return "Information not found in the provided documents."

        context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        # The prompt remains the same, as it's model-agnostic
        full_prompt = (
            "You are an expert AI assistant for analyzing policy documents. Your task is to answer the user's question accurately and concisely, "
            "based ONLY on the provided context. Do not use any external knowledge or make assumptions. "
            "If the answer is not present in the context, you must state that the information is not available in the provided documents.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{question}"
        )
        
        logger.info(f"Sending prompt to Google Gemini for question: '{question}'")
        
        # --- MODIFIED LLM CALL for Google Gemini ---
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=full_prompt
        )

        # --- END MODIFIED LLM CALL ---
        
        # The response object is simpler to access
        return response.text.strip()

    except Exception as e:
        logger.error(f"Error during Gemini RAG for question '{question}': {e}", exc_info=True)
        return "An error occurred while generating the answer from the AI model."

# --- API Endpoint ---
@app.post("/hackrx/run", 
          response_model=HackRxResponse, 
          summary="Run Hackathon Submission",
          dependencies=[Depends(verify_token)])
async def run_submission(payload: HackRxRequest):
    vector_store = process_document_from_url(str(payload.documents))
    loop = asyncio.get_running_loop()

    tasks = []
    answers = []
    for question in payload.questions:
        task = loop.run_in_executor(
            executor, 
            answer_question_with_rag, # The function to run
            question,                 # The first argument to the function
            vector_store              # The second argument to the function
        )
        tasks.append(task)

    logger.info(f"Starting {len(tasks)} RAG queries in parallel...")
    answers = await asyncio.gather(*tasks)
    logger.info("All RAG queries have completed.")

    return HackRxResponse(answers=answers)

# --- Main Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting HackRx RAG API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)