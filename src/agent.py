"""
Capa 3: Agente RAG conectado a Groq.

Toma la pregunta, recupera los chunks más relevantes (capa 2) y se los pasa
a Groq junto con instrucciones de responder solo con ese contexto.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from vectorstore import load_vectorstore

load_dotenv()

SYSTEM_PROMPT = """Eres un asistente interno de Santos Pegasus Soluciones.
Responde ÚNICAMENTE con base en el CONTEXTO proporcionado. Si la respuesta no
está en el contexto, di claramente que no tienes esa información en los
documentos disponibles — no inventes datos.
Responde en español, de forma clara y directa."""


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


def build_context(chunks) -> str:
    """Concatena los fragmentos recuperados en un bloque de texto para el prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.metadata.get("source", "desconocido")
        parts.append(f"[Fragmento {i} - fuente: {source}]\n{c.page_content}")
    return "\n\n".join(parts)


def ask(question: str, k: int = 4) -> dict:
    """
    Función principal del agente: retrieval + generación.
    Devuelve la respuesta junto con las fuentes usadas (para el README y para
    que el usuario final pueda verificar de dónde salió la respuesta).
    """
    vectorstore = load_vectorstore()
    relevant_chunks = vectorstore.similarity_search(question, k=k)
    context = build_context(relevant_chunks)

    client = get_client()
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}",
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    sources = sorted({c.metadata.get("source", "?") for c in relevant_chunks})

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    # Prueba de la capa completa por CLI, antes de envolverla en FastAPI.
    print("Agente RAG listo. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        result = ask(q)
        print(f"\nRespuesta: {result['answer']}")
        print(f"Fuentes: {result['sources']}\n")