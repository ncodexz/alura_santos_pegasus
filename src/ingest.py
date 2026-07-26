"""
Capa 1: Ingesta de documentos.

Carga los PDFs de data/ y los trocea en fragmentos (~800 caracteres, con
solape de 120) para que puedan convertirse en embeddings en la capa 2.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"


def load_documents(data_dir: Path = DATA_DIR):
    """
    Carga todos los PDFs de la carpeta data/ y devuelve una lista de objetos
    Document de LangChain (cada uno con .page_content y .metadata, incluyendo
    el nombre del archivo de origen — útil para citar la fuente en las respuestas).
    """
    documents = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron PDFs en {data_dir}. "
            "Coloca los 5 PDFs de Santos Pegasus ahí antes de continuar."
        )

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()  # una entrada por página
        documents.extend(docs)
        print(f"  Cargado: {pdf_path.name} ({len(docs)} páginas)")

    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    Trocea los documentos en fragmentos más pequeños.

    chunk_size=800: suficientemente grande para mantener contexto (una idea completa),
    suficientemente pequeño para que el retrieval sea preciso (no traer medio documento).

    chunk_overlap=120: los fragmentos se solapan un poco para no cortar una idea justo
    a la mitad entre dos chunks (ej. una frase que empieza en el chunk 3 y termina en el 4).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Prueba rápida y aislada de esta capa, sin depender de vectorstore.py ni del LLM.
    print("Cargando documentos...")
    docs = load_documents()
    print(f"\nTotal páginas cargadas: {len(docs)}")

    chunks = split_documents(docs)
    print(f"Total chunks generados: {len(chunks)}")
    print("\n--- Ejemplo del primer chunk ---")
    print(chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)