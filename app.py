import os
import io
import requests
import PyPDF2
from docx import Document
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import google.generativeai as genai  # NEW: Import the Google GenAI library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangchainDocument

# Initialize FastAPI app
app = FastAPI(title="LLM-Powered Intelligent Query-Retrieval System")

# --- Environment Variables ---
# Use the new environment variable name for the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the key is available
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the application.")

# Initialize Google GenAI client
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize a free, open-source embeddings model for local use
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Pydantic Models for API Request and Response ---
class QuestionRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QuestionResponse(BaseModel):
    answers: List[str]

# --- Helper Functions for Document Handling ---
def download_and_parse_document(document_url: str):
    """Downloads a document and extracts text based on its file extension."""
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        content = response.content
        parsed_url = urlparse(document_url)
        filename = os.path.basename(parsed_url.path).lower()
        
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "".join([page.extract_text() or "" for page in reader.pages])
            return text
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = "".join([para.text + "\n" for para in doc.paragraphs])
            return text
        else:
            raise ValueError("Unsupported document format. Only PDF and DOCX are supported.")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {type(e).__name__} - {e}")


# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=QuestionResponse)
async def run_query_retrieval(request: QuestionRequest):
    """
    Main endpoint to process documents, retrieve relevant clauses,
    and answer questions using an LLM.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No document URLs provided.")
    
    # 1. Document Ingestion and Preprocessing for ALL documents
    all_docs_text = ""
    for doc_url in request.documents:
        all_docs_text += download_and_parse_document(doc_url)
        all_docs_text += "\n\n---END-OF-DOCUMENT---\n\n"
    
    # Split the combined text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    docs = [LangchainDocument(page_content=chunk) for chunk in text_splitter.split_text(all_docs_text)]
    
    # 2. Create local FAISS vector store
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {type(e).__name__} - {e}")
    
    answers = []
    
    for question in request.questions:
        # 3. Perform semantic search (Clause Matching)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(question)
        retrieved_context = " ".join([doc.page_content for doc in relevant_docs])
        
        # 4. LLM Generation using Google Gemini
        prompt = f"""
        You are an expert in policy documents. Your task is to answer a user's question
        based on the provided context from a policy document. If the answer is not
        explicitly stated in the context, state that you cannot find the information.

        Question: {question}

        Context:
        {retrieved_context}

        Answer:
        """
        
        try:
            # NEW: Call the Google GenAI model instead of OpenAI
            response = llm_model.generate_content(prompt)
            answer_text = response.text.strip()
            answers.append(answer_text)
        except Exception as e:
            answers.append(f"An error occurred while generating the answer: {type(e).__name__} - {e}")
            
    # 5. JSON Output
    return QuestionResponse(answers=answers)