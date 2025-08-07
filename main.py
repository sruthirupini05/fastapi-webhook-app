from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pinecone
from openai import OpenAI
import psycopg2
from typing import List, Dict

# NEW IMPORTS
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# NEW: Serve static files (HTML, CSS, JS, favicon.ico) from the 'static' directory
# This fixes the /favicon.ico 404 error
app.mount("/static", StaticFiles(directory="static"), name="static")

# NEW: Create a root endpoint that serves your index.html file
# This fixes the / 404 error
@app.get("/", response_class=FileResponse)
async def read_root():
    return "static/index.html"


# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="your-pinecone-env")
index = pinecone.Index("your-index-name")

# Initialize OpenAI
openai_client = OpenAI(api_key="your-openai-api-key")

# PostgreSQL connection
try:
    conn = psycopg2.connect(
        dbname="your_db",
        user="your_user",
        password="your_password",
        host="your_host"
    )
    cur = conn.cursor()
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")
    # You might want to exit or handle this more gracefully
    conn = None
    cur = None

class QueryRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def extract_embeddings(text: str) -> List[float]:
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def store_in_pinecone(doc_id: str, embedding: List[float]):
    index.upsert(vectors=[(doc_id, embedding)])

def retrieve_from_pinecone(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result.matches

def query_llm(question: str, context: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

def save_to_postgres(doc_id: str, answer: str):
    if conn and cur:
        try:
            cur.execute("INSERT INTO responses (doc_id, answer) VALUES (%s, %s)", (doc_id, answer))
            conn.commit()
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
    else:
        print("Database connection is not available.")

@app.post("/hackrx/run")
async def process_query(request: QueryRequest):
    answers = []
    # Simplified logic for this example. The original was a bit complex.
    # I've moved the embedding extraction for questions to be outside the inner loop.
    for question in request.questions:
        question_embedding = extract_embeddings(question)
        context_results = retrieve_from_pinecone(question_embedding)
        
        # A more robust way to get context text
        context = " ".join([
            (match.metadata.get("text", "") or match.id) for match in context_results if match and match.metadata
        ])
        
        answer = query_llm(question, context)
        answers.append(answer)
        
        # You'll need to decide which doc_id to associate with the answer.
        # For simplicity, let's use the first document URL from the request.
        if request.documents:
            save_to_postgres(request.documents[0], answer)
    
    return QueryResponse(answers=answers)


if __name__ == "__main__":
    import uvicorn
    # This ensures your app runs and your new endpoints are active
    uvicorn.run(app, host="0.0.0.0", port=8000)