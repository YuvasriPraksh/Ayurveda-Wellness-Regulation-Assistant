# Ayurveda & Wellness Regulation Assistant

An AI-powered, RAG-based assistant designed to help Ayurveda manufacturers, startups, and researchers understand AYUSH licensing, Ayurvedic product labeling, IP, and traditional formulation regulations.

## 🌟 Features
- **Grounded Retrieval**: Uses FAISS and LangChain to fetch relevant regulatory guidelines.
- **Strict Fallback (Mandatory Feature)**: If the query cannot be answered based on the available sources, the assistant strictly refuses to answer, ensuring 0% hallucinations.
- **Polished UI**: Built with Streamlit, customized for a professional and wellness-oriented look.
- **Local Embeddings**: Fast document processing using `sentence-transformers`.
- **Citations**: Automatically appends the retrieved sources to the answer.

## 🛠 Architecture
1. **Frontend**: Streamlit
2. **LLM**: Google Gemini 1.5 Flash (via `langchain-google-genai`)
3. **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
4. **Vector Store**: FAISS (local)
5. **Orchestration**: LangChain

## 🚀 Quick Start (1-Minute Setup)

### 1. Install Dependencies
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file and add your API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Build the Knowledge Base
Run the document processor to chunk the documents in `data/` and build the FAISS index:
```bash
python src/document_processor.py
```

### 4. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure
```text
.
├── app.py                      # Main Streamlit application
├── data/                       # Directory for regulatory PDFs and text files
│   └── sample_regulation.txt   # Sample AYUSH guidelines
├── src/
│   ├── document_processor.py   # Script to process documents and build FAISS vector DB
│   └── rag_pipeline.py         # RAG logic (similarity search, strict prompt, LLM call)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # Project documentation
```

## 🏆 Hackathon Priorities Addressed
- **Visual/UX Quality**: Clean, themed Streamlit UI with sidebars and avatars.
- **Complete RAG Pipeline**: End-to-end ingestion, embedding, and retrieval generation.
- **Real-world Relevance**: Solves a real problem for Ayurvedic startups navigating complex regulations.
- **End-to-End Reliability**: Graceful error handling and mandatory fallback implementation.
