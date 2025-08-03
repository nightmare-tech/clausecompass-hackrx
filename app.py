import logging
import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import requests

# --- Core AI/ML & Data Libraries ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from google import genai
from flashrank import Ranker

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment and Global Initializations ---
load_dotenv()
executor = ThreadPoolExecutor(max_workers=os.cpu_count())
app = FastAPI(title="ClauseCompass - HackRx 6.0 RAG API", version="3.0.0-caching")

# --- CACHING STRATEGY ---
# Global in-memory cache to store pre-processed FAISS vector stores.
# The key will be the document's base URL, and the value will be the FAISS object.
VECTOR_STORE_CACHE: Dict[str, FAISS] = {}

# --- Initialize Global Models ---
# ... (Embedding model, Re-ranker, and Gemini client initializations remain the same) ...
# (Using the fast, corrected versions from the previous answer)
EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
client = genai.Client()


# --- API Security & Pydantic Models ---
EXPECTED_BEARER_TOKEN = "e66c2e8eb6884ded2c7177421784e760b34b9297bfebc20a2a272cc63357270d"

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

async def verify_token(Authorization: str = Header(..., description="Bearer token for authentication.")):
    if Authorization != f"Bearer {EXPECTED_BEARER_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing auth token.")

# --- Core Logic Functions ---

def process_document_from_url(doc_url: str) -> FAISS:
    """The slow processing function that will only be called on a cache miss."""
    logger.info(f"--- CACHE MISS --- Starting expensive processing for: {doc_url.split('?')[0]}")
    # ... (The optimized, batched processing function remains exactly the same) ...
    tmp_file_path = None
    try:
        response = requests.get(doc_url, stream=True)
        response.raise_for_status()
        file_extension = Path(doc_url.split('?')[0]).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        loader = UnstructuredFileLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract content.")
            
        chunk_texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(chunk_texts, embedding_model, metadatas=[chunk.metadata for chunk in chunks])
        return vector_store
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

from flashrank import RerankRequest
def answer_question_with_rag(question: str, vector_store: FAISS) -> str:
    """Answers a question using the RAG pipeline. This function is fast."""
    try:
        retrieved_docs = vector_store.similarity_search(question, k=20)
        if not retrieved_docs:
            return "Information not found in the provided documents."
        # passages_for_reranker = [
        #     {"id": i, "text": doc.page_content} 
        #     for i, doc in enumerate(retrieved_docs)
        # ]
        # request = RerankRequest(
        #     query=question,
        #     passages=passages_for_reranker,
        # )
        # reranked_results = reranker.rerank(request)
        # top_k_reranked = reranked_results[:10]
        context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        full_prompt = (
            "You are an expert AI assistant. Based ONLY on the CONTEXT provided, give a clear and concise answer to the QUESTION. "
            "If the answer is not in the CONTEXT, state 'The information is not available in the provided document.'\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=full_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error during RAG for question '{question}': {e}", exc_info=True)
        return "An error occurred while generating the answer from the AI model."

async def get_or_create_vector_store(doc_url: str) -> FAISS:
    """
    Caching wrapper. Gets a vector store from cache or creates it if it doesn't exist.
    """
    # Create a stable cache key from the base URL, ignoring temporary query params.
    parsed_url = urlparse(doc_url)
    cache_key = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    if cache_key in VECTOR_STORE_CACHE:
        logger.info(f"--- CACHE HIT --- Found pre-processed vector store for: {cache_key}")
        return VECTOR_STORE_CACHE[cache_key]
    else:
        loop = asyncio.get_running_loop()
        # Run the blocking IO/CPU-bound function in a thread pool.
        vector_store = await loop.run_in_executor(
            executor, process_document_from_url, doc_url
        )
        # Save the newly created store to the cache for next time.
        VECTOR_STORE_CACHE[cache_key] = vector_store
        return vector_store

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, summary="Run Submission", dependencies=[Depends(verify_token)])
async def run_submission(payload: HackRxRequest):
    """Main API endpoint that now uses the caching layer."""
    try:
        # Get the vector store using our new caching function.
        # The first call for a document will be slow; subsequent calls will be fast.
        vector_store = await get_or_create_vector_store(str(payload.documents))
        
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(executor, answer_question_with_rag, q, vector_store)
            for q in payload.questions
        ]
        logger.info(f"Starting {len(tasks)} RAG queries in parallel...")
        final_answers: List[str] = await asyncio.gather(*tasks)
        logger.info("All RAG queries have completed.")
        return HackRxResponse(answers=final_answers)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

# --- Main Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ClauseCompass RAG API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)