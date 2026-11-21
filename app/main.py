from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import Assistant, Retriever, load_chunks

app = FastAPI(title="RAG Docs Assistant")
assistant = Assistant(Retriever(load_chunks()))


class Question(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "mode": "llm" if assistant.llm else "extractive"}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return assistant.ask(q.question, k=q.k)
