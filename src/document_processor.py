import os
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_db"
DATA_DIR = "data"

def get_embeddings():
    # Using sentence-transformers for fast, reliable local embeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    print("Loading documents...")
    documents = []
    
    # Load Text Files
    for text_file in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        loader = TextLoader(text_file, encoding='utf-8')
        documents.extend(loader.load())
        
    # Load PDF Files
    for pdf_file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())

    if not documents:
        print("No documents found in the data/ directory.")
        return None

    print(f"Loaded {len(documents)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print(f"Created {len(docs)} chunks. Building FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the database locally
    vectorstore.save_local(DB_DIR)
    print(f"FAISS index built and saved to {DB_DIR}/")
    return vectorstore

def load_vector_store():
    if not os.path.exists(DB_DIR):
        print(f"Database directory {DB_DIR} not found. Building a new one...")
        return build_vector_store()
    
    embeddings = get_embeddings()
    # allow_dangerous_deserialization=True is required for recent langchain FAISS versions
    return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    build_vector_store()
