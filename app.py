import logging
import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import hashlib

from fastapi import FastAPI, Depends, HTTPException, status, Header
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
import requests

# --- Core AI/ML & Data Libraries ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from flashrank import Ranker, RerankRequest

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment and Global Initializations ---
load_dotenv()
executor = ThreadPoolExecutor(max_workers=os.cpu_count())
app = FastAPI(title="ClauseCompass - HackRx 6.0 RAG API", version="13.0.0-final-final")

# --- PERSISTENT CACHING STRATEGY ---
CACHE_DIR = Path(os.getenv("CACHE_PATH", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_CACHE: Dict[str, FAISS] = {}

# --- Initialize Global Models ---
try:
    logger.info("Initializing models...")
    EMBEDDING_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    jamba_client = AI21Client(api_key=os.environ.get("AI21_API_KEY"))
    logger.info("All models initialized successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize models. Error: {e}", exc_info=True)
    raise

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
    logger.info(f"--- SLOW PATH --- Starting expensive processing for: {doc_url.split('?')[0]}")
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract content.")
            
        chunk_texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(chunk_texts, embedding_model, metadatas=[chunk.metadata for chunk in chunks])
        return vector_store
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def answer_question_with_rag(question: str, vector_store: FAISS) -> str:
    """Answers a question using the RAG pipeline with the correct Jamba Chat API call."""
    try:
        retrieved_docs = vector_store.similarity_search(question, k=20)
        if not retrieved_docs:
            return "Information not found in the provided documents."
        
        passages_for_reranker = [{"id": i, "text": doc.page_content} for i, doc in enumerate(retrieved_docs)]
        
        request = RerankRequest(query=question, passages=passages_for_reranker)
        reranked_results = reranker.rerank(request)

        top_k_reranked = reranked_results[:5]
        context_str = "\n---\n".join([p['text'] for p in top_k_reranked])
        
        system_prompt = (
            "You are an expert AI assistant. Based ONLY on the CONTEXT provided by the user, "
            "give a clear and concise answer to the user's QUESTION. "
            "If the answer is not in the CONTEXT, state 'The information is not available in the provided document.'"
        )
        user_prompt = (
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION:\n{question}"
        )
        
        # The AI21 ChatMessage object requires BOTH 'text' and 'content' to satisfy its validator.
        messages = [
            ChatMessage(role="system", text=system_prompt, content=system_prompt),
            ChatMessage(role="user", text=user_prompt, content=user_prompt)
        ]
        
        logger.info(f"Sending prompt to Jamba Chat API for question: '{question}'")
        
        response = jamba_client.chat.completions.create(
            model="jamba-large-1.7",
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            top_p=1
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error during RAG for question '{question}': {e}", exc_info=True)
        return "An error occurred while generating the answer from the AI model."


async def get_or_create_vector_store(doc_url: str) -> FAISS:
    """Caching wrapper: Checks RAM, then Disk, then processes if no cache is found."""
    parsed_url = urlparse(doc_url)
    cache_key = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    if cache_key in VECTOR_STORE_CACHE:
        logger.info(f"--- RAM CACHE HIT --- for: {cache_key}")
        return VECTOR_STORE_CACHE[cache_key]

    safe_filename = hashlib.sha256(cache_key.encode()).hexdigest()
    disk_cache_path = CACHE_DIR / safe_filename

    if disk_cache_path.exists():
        logger.info(f"--- DISK CACHE HIT --- Loading from {disk_cache_path}...")
        vector_store = FAISS.load_local(
            folder_path=str(disk_cache_path), 
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        VECTOR_STORE_CACHE[cache_key] = vector_store
        return vector_store

    loop = asyncio.get_running_loop()
    vector_store = await loop.run_in_executor(executor, process_document_from_url, doc_url)
    
    logger.info(f"--- SAVING TO CACHE --- Saving to disk at {disk_cache_path}")
    vector_store.save_local(folder_path=str(disk_cache_path))
    VECTOR_STORE_CACHE[cache_key] = vector_store
    
    return vector_store


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, summary="Run Submission", dependencies=[Depends(verify_token)])
async def run_submission(payload: HackRxRequest):
    """Main API endpoint that uses the robust caching layer."""
    try:
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


# --- Main Guard for Local Development ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ClauseCompass RAG API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)