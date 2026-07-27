"""
Capa 3: Agente RAG conectado a Groq.

Toma la pregunta, recupera los chunks más relevantes (vía Pinecone, capa 2)
y se los pasa a Groq junto con instrucciones de responder solo con ese
contexto.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from vectorstore import load_vectorstore, search

load_dotenv()

SYSTEM_PROMPT = """Eres un asistente interno de Santos Pegasus Soluciones.
Responde ÚNICAMENTE con base en el CONTEXTO proporcionado. Si la respuesta no
está en el contexto, di claramente que no tienes esa información en los
documentos disponibles — no inventes datos.
Responde en español, de forma clara y directa."""

# Cache global: la conexión al índice de Pinecone se abre una sola vez.
_index = None


def get_index():
    global _index
    if _index is None:
        _index = load_vectorstore()
    return _index


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )


def build_context(hits) -> str:
    """Concatena los fragmentos recuperados en un bloque de texto para el prompt."""
    parts = []
    for i, hit in enumerate(hits, 1):
        fields = hit["fields"]
        source = fields.get("source", "desconocido")
        text = fields.get("chunk_text", "")
        parts.append(f"[Fragmento {i} - fuente: {source}]\n{text}")
    return "\n\n".join(parts)


def ask(question: str, k: int = 4) -> dict:
    """
    Función principal del agente: retrieval (Pinecone) + generación (Groq).
    Devuelve la respuesta junto con las fuentes usadas.
    """
    index = get_index()
    hits = search(index, question, k=k)
    context = build_context(hits)

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
    sources = sorted({hit["fields"].get("source", "?") for hit in hits})

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    print("Agente RAG listo. Escribe 'salir' para terminar.\n")
    while True:
        q = input("Pregunta: ").strip()
        if q.lower() in ("salir", "exit", "quit"):
            break
        result = ask(q)
        print(f"\nRespuesta: {result['answer']}")
        print(f"Fuentes: {result['sources']}\n")