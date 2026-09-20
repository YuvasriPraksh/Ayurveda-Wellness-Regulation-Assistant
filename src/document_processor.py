import os
import glob
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Directories
BASE_DATA_DIR = "data"
DOCUMENTS_SUBDIR = "documents"
DATA_DIR = os.path.join(BASE_DATA_DIR, DOCUMENTS_SUBDIR)
DB_DIR = "faiss_db"
META_FILE = os.path.join(DB_DIR, "index_meta.json")

def get_embeddings():
    """Return the embedding model used for all documents."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def _file_hash(path: str) -> str:
    """Compute a short SHA256 hash of a file's contents for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def _load_index_metadata() -> Dict[str, Any]:
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_index_metadata(meta: Dict[str, Any]):
    os.makedirs(DB_DIR, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def clear_index():
    """Remove the existing FAISS index and metadata.

    Use this when you want to force a full re‑index of the knowledge base.
    """
    if os.path.isdir(DB_DIR):
        for root, dirs, files in os.walk(DB_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(DB_DIR)
    if os.path.exists(META_FILE):
        os.remove(META_FILE)
    print("FAISS index and metadata cleared.")

def build_vector_store(force: bool = False):
    """Build (or rebuild) the FAISS vector store.

    The function automatically discovers *.pdf and *.txt files under ``data/documents``.
    It preserves metadata such as source filename, page number (for PDFs), title, and a simple
    section placeholder when detectable. Duplicate indexing is avoided by comparing file hashes
    with the stored metadata unless ``force`` is True.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Gather current file hashes
    current_files: Dict[str, str] = {}
    for ext in ("*.pdf", "*.txt"):
        for path in glob.glob(os.path.join(DATA_DIR, ext)):
            current_files[path] = _file_hash(path)

    # Load previous metadata
    meta = _load_index_metadata()
    previous_hashes: Dict[str, str] = meta.get("file_hashes", {})

    if not force and previous_hashes == current_files and os.path.isdir(DB_DIR):
        print("No changes detected in the knowledge base – loading existing index.")
        return load_vector_store()

    print("Building FAISS index – processing documents...")
    documents = []

    # Load TXT files
    for txt_path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(txt_path)
            doc.metadata["page"] = "Not available"
            # Simple section detection: first line as heading if it looks like a title
            first_line = doc.page_content.splitlines()[0] if doc.page_content else ""
            if len(first_line.split()) <= 10:
                doc.metadata["section"] = first_line.strip()
            else:
                doc.metadata["section"] = ""
        documents.extend(docs)

    # Load PDF files – each page becomes a separate document
    for pdf_path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        filename = os.path.basename(pdf_path)
        for doc in docs:
            doc.metadata["source"] = filename
            # PyPDFLoader provides a 0‑based "page" field; make it 1‑based for UI friendliness
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1
            else:
                doc.metadata["page"] = "Not available"
            # Title fallback – filename without extension
            doc.metadata.setdefault("title", os.path.splitext(filename)[0])
            # Section placeholder – attempt to pull heading from the page text (first line)
            first_line = doc.page_content.splitlines()[0] if doc.page_content else ""
            if len(first_line.split()) <= 12:
                doc.metadata["section"] = first_line.strip()
            else:
                doc.metadata["section"] = ""
        documents.extend(docs)

    if not documents:
        print("No documents found in the data/documents directory.")
        return None

    print(f"Loaded {len(documents)} raw documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks. Building FAISS index...")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_DIR)

    # Update metadata
    meta = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "file_hashes": current_files,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vectorstore_path": DB_DIR,
    }
    _save_index_metadata(meta)
    print(f"FAISS index saved to {DB_DIR}/ and metadata written to {META_FILE}.")
    return vectorstore

def load_vector_store():
    """Load the persisted FAISS index. If it does not exist, build it automatically."""
    if not os.path.isdir(DB_DIR):
        print(f"FAISS directory {DB_DIR} not found – building a new index.")
        return build_vector_store()
    embeddings = get_embeddings()
    # ``allow_dangerous_deserialization`` required for recent LangChain versions
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

def get_kb_status() -> Dict[str, Any]:
    """Return a quick status report for the knowledge base.

    Example output::
        {
            "documents": 3,
            "chunks": 152,
            "embedding_model": "all-MiniLM-L6-v2",
            "vectorstore": "faiss_db",
            "last_indexed": "2026-09-20T11:02:15Z"
        }
    """
    meta = _load_index_metadata()
    if not meta:
        return {
            "documents": 0,
            "chunks": 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "vectorstore": DB_DIR,
            "last_indexed": None,
        }
    return {
        "documents": meta.get("num_documents", 0),
        "chunks": meta.get("num_chunks", 0),
        "embedding_model": meta.get("embedding_model", "all-MiniLM-L6-v2"),
        "vectorstore": meta.get("vectorstore_path", DB_DIR),
        "last_indexed": meta.get("built_at"),
    }

if __name__ == "__main__":
    # Simple CLI to build, clear or show status
    import argparse

    parser = argparse.ArgumentParser(description="Document processor for AYURA REGULATE")
    parser.add_argument("action", choices=["build", "clear", "status"], help="Action to perform")
    args = parser.parse_args()

    if args.action == "build":
        build_vector_store(force=True)
    elif args.action == "clear":
        clear_index()
    elif args.action == "status":
        print(json.dumps(get_kb_status(), indent=2))
