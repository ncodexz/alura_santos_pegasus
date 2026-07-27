"""
Capa 2: Índice vectorial (Pinecone, con embeddings integrados).

Pinecone genera los embeddings en sus propios servidores (modelo
"multilingual-e5-large") y guarda + busca los vectores — nuestro servidor
nunca carga ningún modelo pesado. Esto es lo que evita el problema de
memoria que teníamos con FAISS + embeddings locales en el free tier de
Render.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "alura-agente-santos-pegasus"
NAMESPACE = "santos-pegasus"


def get_client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_or_create_index():
    pc = get_client()
    if not pc.has_index(INDEX_NAME):
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "multilingual-e5-large",
                "field_map": {"text": "chunk_text"},
            },
        )
    index_config = pc.describe_index(INDEX_NAME)
    return pc.Index(host=index_config.host)


def build_vectorstore(chunks):
    """Sube los chunks a Pinecone (crea el índice si no existe)."""
    index = get_or_create_index()

    records = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "desconocido")
        records.append(
            {
                "_id": f"chunk-{i}",
                "chunk_text": chunk.page_content,
                "source": source,
            }
        )

    # Pinecone recomienda subir en lotes pequeños (~90 registros) por request.
    batch_size = 90
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert_records(namespace=NAMESPACE, records=batch)
        print(f"  Subidos {i + len(batch)}/{len(records)} chunks")

    return index


def load_vectorstore():
    """Conecta al índice ya existente (no vuelve a subir nada)."""
    return get_or_create_index()


def search(index, query: str, k: int = 4):
    """Busca los k fragmentos más relevantes para una pregunta."""
    response = index.search(
        namespace=NAMESPACE,
        query={"inputs": {"text": query}, "top_k": k},
        fields=["chunk_text", "source"],
    )
    return response["result"]["hits"]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import load_documents, split_documents

    print("Cargando y troceando documentos...")
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Subiendo {len(chunks)} chunks a Pinecone (embeddings integrados)...")
    idx = build_vectorstore(chunks)

    print("\nEsperando unos segundos a que Pinecone indexe...")
    import time
    time.sleep(10)

    test_queries = [
        "¿Cuál es el protocolo de respuesta a incidentes?",
        "¿Qué stack se usa en el back-end?",
    ]

    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        hits = search(idx, q, k=2)
        for hit in hits:
            fields = hit["fields"]
            print(f"[{fields.get('source', '?')}] {fields.get('chunk_text', '')[:200]}...")