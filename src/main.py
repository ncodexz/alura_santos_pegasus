"""
Capa 4: API HTTP.

Envuelve agent.py en un endpoint FastAPI (POST /ask) para poder desplegar
el agente en Render y que sea accesible públicamente.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from agent import ask

app = FastAPI(title="Alura Agente - Santos Pegasus Soluciones")


class Question(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Alura Agente - Santos Pegasus Soluciones"}


@app.post("/ask")
def ask_agent(payload: Question):
    result = ask(payload.question)
    return result