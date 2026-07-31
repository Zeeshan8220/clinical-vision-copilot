"""FastAPI endpoint for the Knowledge/RAG Agent."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from rag_agent import answer_question

app = FastAPI(title="Clinical Vision Copilot — Knowledge/RAG Agent API")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    result = answer_question(payload.question)
    result["disclaimer"] = (
        "Answers are generated from a small curated reference set for "
        "educational/demo purposes and are not a substitute for consulting "
        "primary clinical guidelines or a licensed physician."
    )
    return result
