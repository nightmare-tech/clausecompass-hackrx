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
from groq import Groq
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Initialize Global Objects ---
app = FastAPI(
    title="HackRx 6.0 RAG API - ClauseCompass",
    description="An LLM-Powered Intelligent Query–Retrieval System for Bajaj Finserv Health."
)

# Load Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")
groq_client = Groq(api_key=GROQ_API_KEY)
LLAMA_MODEL_NAME = "llama3-8b-8192" # Or "llama3-70b-8192" for higher quality

# Load Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    # This is the original model object
    # core_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME) # This can be removed if not used elsewhere
    
    # This is the NEW LangChain-compatible wrapper object
    # It will use the sentence-transformers library under the hood
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Specify CPU for consistency
    )
    
    logger.info(f"Successfully loaded LangChain-wrapped embedding model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}", exc_info=True)
    raise

# Hardcoded Bearer Token for the hackathon
EXPECTED_BEARER_TOKEN = "e66c2e8eb6884ded2c7177421784e760b34b9297bfebc20a2a272cc63357270d"

# --- Pydantic Models for API Schema ---
class HackRxRequest(BaseModel):
    documents: HttpUrl # Pydantic validates that this is a valid URL string
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Authentication Dependency ---
async def verify_token(Authorization: str = Header(..., description="Bearer token for authentication.")):
    """A simple dependency to check for the hardcoded bearer token."""
    if Authorization != f"Bearer {EXPECTED_BEARER_TOKEN}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
        )

# --- Core RAG Logic Functions ---
def process_document_from_url(doc_url: str) -> FAISS:
    """
    Downloads a document from a URL (PDF, DOCX, EML, TXT), loads, chunks, 
    and indexes it into an in-memory FAISS vector store.
    """
    tmp_file_path = None
    try:
        logger.info(f"Downloading document from URL: {doc_url}")
        response = requests.get(doc_url, stream=True)
        response.raise_for_status()

        # Get file extension from URL path for tempfile
        file_extension = Path(doc_url.split('?')[0]).suffix or ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Document saved to temporary file: {tmp_file_path}")

        # Use UnstructuredFileLoader to handle various document types
        loader = UnstructuredFileLoader(tmp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        if not chunks:
            logger.warning(f"No text could be extracted from the document at {doc_url}")
            raise HTTPException(status_code=400, detail="Could not extract content from the provided document.")

        logger.info(f"Document chunked into {len(chunks)} pieces. Creating FAISS index...")
        
        # Create the in-memory FAISS index from the chunks
        vector_store = FAISS.from_documents(chunks, embedding_model)
        
        logger.info("In-memory FAISS index created successfully.")
        
        return vector_store

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=400, detail="Could not download or access the document URL.")
    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the document.")
    finally:
        # Clean up the temporary file from disk
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def answer_question_with_rag(question: str, vector_store: FAISS) -> str:
    """Performs a RAG query to answer a single question using Groq."""
    try:
        logger.info(f"Performing similarity search for question: '{question}'")
        retrieved_docs = vector_store.similarity_search(question, k=4) # Retrieve top 4 relevant chunks
        
        if not retrieved_docs:
            return "Information not found in the provided documents."

        context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        system_prompt = (
            "You are an expert AI assistant for analyzing policy documents. Your task is to answer the user's question accurately and concisely, "
            "based ONLY on the provided context. Do not use any external knowledge or make assumptions. "
            "If the answer is not present in the context, you must state that the information is not available in the provided documents.\n\n"
            f"CONTEXT:\n{context_str}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        logger.info(f"Sending prompt to Groq/Llama for question: '{question}'")
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=LLAMA_MODEL_NAME,
            max_tokens=512, # Increased slightly for more detailed answers
            temperature=0.1 # Keep it factual
        )
        
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error during Groq RAG for question '{question}': {e}", exc_info=True)
        return "An error occurred while generating the answer from the AI model."

# --- API Endpoint ---

@app.post("/hackrx/run", 
          response_model=HackRxResponse, 
          summary="Run Hackathon Submission",
          dependencies=[Depends(verify_token)])
async def run_submission(payload: HackRxRequest):
    """
    Main endpoint for the hackathon. It takes a document URL and a list of questions,
    performs on-the-fly RAG using a FAISS in-memory index and Groq's Llama3,
    and returns a list of answers.
    """
    # 1. Process the document from the URL into a FAISS vector store
    vector_store = process_document_from_url(str(payload.documents))
    
    # 2. Loop through each question and generate an answer using RAG
    answers = []
    for question in payload.questions:
        answer = answer_question_with_rag(question, vector_store)
        answers.append(answer)
        
    # 3. Return the final structured response
    return HackRxResponse(answers=answers)

# --- Main Guard for running the app ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting HackRx RAG API server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)