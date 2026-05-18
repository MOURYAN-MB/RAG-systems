from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split(data_dir: Path = DATA_DIR) -> list:
    docs = []

    # PDFs: each page becomes a Document with 'page' and 'source' in metadata
    pdf_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs.extend(pdf_loader.load())

    # Plain text and markdown files
    for glob_pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(data_dir),
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs.extend(loader.load())

    unique_sources = set(d.metadata.get("source", "") for d in docs)
    print(f"Loaded {len(docs)} pages across {len(unique_sources)} file(s)")

    # RecursiveCharacterTextSplitter tries larger separators first (\n\n, \n, ". ")
    # before falling back to character splits — preserves sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks