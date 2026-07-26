"""
Capa 2: Vector store + retrieval.

Convierte los chunks en embeddings y los indexa en FAISS para poder buscar,
dada una pregunta, los fragmentos más relevantes por similitud semántica.

Los embeddings son locales (HuggingFace/sentence-transformers), no de Groq —
Groq solo se usa para generar la respuesta final (ver agent.py).
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"

# Modelo pequeño y rápido, corre en CPU sin problema, buena calidad para español.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks, save: bool = True):
    """Crea el índice FAISS a partir de los chunks y opcionalmente lo guarda en disco."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if save:
        INDEX_DIR.mkdir(exist_ok=True)
        vectorstore.save_local(str(INDEX_DIR))
        print(f"Índice guardado en {INDEX_DIR}")

    return vectorstore


def load_vectorstore():
    """Carga un índice FAISS ya guardado (evita re-procesar los PDFs cada vez)."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    # Aseguramos que esta carpeta esté en sys.path, sin importar cómo se invocó
    # el script (algunos entornos, como Render, no la añaden automáticamente).
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Prueba aislada: construye el índice desde cero y lanza un par de queries manuales
    # para confirmar que el retrieval trae fragmentos relevantes ANTES de conectar el LLM.
    from ingest import load_documents, split_documents

    print("Cargando y troceando documentos...")
    docs = load_documents()
    chunks = split_documents(docs)

    print("Generando embeddings e indexando (puede tardar un poco la primera vez)...")
    vs = build_vectorstore(chunks)

    test_queries = [
        "¿Cuál es el protocolo de respuesta a incidentes?",
        "¿Qué stack se usa en el back-end?",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        results = vs.similarity_search(q, k=2)
        for r in results:
            source = r.metadata.get("source", "?")
            print(f"[{Path(source).name}] {r.page_content[:200]}...")