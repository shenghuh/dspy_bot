from fastapi import FastAPI

from src.retriever import LocalRetriever
from src.rag_cot import RAGCoT

app       = FastAPI()
retriever = LocalRetriever()
pipeline  = RAGCoT(retriever, k=4)

@app.post("/ask")
async def ask(question: str):
    return {"answer": pipeline(question)}

# Run with:
# python -m uvicorn src.chatbot:app --reload