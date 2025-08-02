from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pinecone
from openai import OpenAI
import psycopg2
from typing import List, Dict

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="your-pinecone-env")
index = pinecone.Index("your-index-name")

# Initialize OpenAI
openai_client = OpenAI(api_key="your-openai-api-key")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="your_db",
    user="your_user",
    password="your_password",
    host="your_host"
)
cur = conn.cursor()

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
    cur.execute("INSERT INTO responses (doc_id, answer) VALUES (%s, %s)", (doc_id, answer))
    conn.commit()

@app.post("/hackrx/run")
async def process_query(request: QueryRequest):
    answers = []
    for doc_url in request.documents:
        # Simulate document processing
        doc_text = f"Content from {doc_url}"  # Placeholder for actual document parsing
        embedding = extract_embeddings(doc_text)
        store_in_pinecone(doc_url, embedding)
        
        for question in request.questions:
            context_results = retrieve_from_pinecone(extract_embeddings(question))
            context = " ".join([match.metadata.get("text", "") for match in context_results])
            answer = query_llm(question, context)
            answers.append(answer)
            save_to_postgres(doc_url, answer)
    
    return QueryResponse(answers=answers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)