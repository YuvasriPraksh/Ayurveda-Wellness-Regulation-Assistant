import os
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.document_processor import load_vector_store

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.1"))

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError("Google API Key not set. Please update the .env file with your GEMINI API Key.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

# --- 1. Intent Classification ---
class IntentClassificationResult(BaseModel):
    intents: List[str] = Field(description="List of identified intents: LICENSING, LABELING, IP, GENERAL_REGULATION, OUT_OF_SCOPE")
    reasoning: str = Field(description="Why these intents were chosen.")

def classify_query(query: str) -> IntentClassificationResult:
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentClassificationResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user's query into one or more of these intents: LICENSING, LABELING, IP, GENERAL_REGULATION, OUT_OF_SCOPE. "
                   "If the query is about Ayurveda regulations, licensing, labeling, IP, or compliance, classify accordingly. "
                   "If the query is completely unrelated (e.g. sports, weather), return OUT_OF_SCOPE."),
        ("human", "{query}")
    ])
    chain = prompt | structured_llm
    return chain.invoke({"query": query})

# --- 2. Retrieval Tool ---
def retrieve_regulations(query: str, k: int = 4) -> List[Dict[str, Any]]:
    vectorstore = load_vector_store()
    if not vectorstore:
        return []

    # Perform similarity search with scores
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Filter by threshold and structure the output
    valid_chunks = []
    for doc, score in results:
        if score <= SIMILARITY_THRESHOLD:
            valid_chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Not available'),
                "score": float(score)
            })
    return valid_chunks

# --- 3. Evidence Verification ---
class VerificationResult(BaseModel):
    supported: bool = Field(description="True if the retrieved documents contain sufficient evidence to answer the query, False otherwise.")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW confidence.")
    reason: str = Field(description="Reason why the evidence is sufficient or insufficient.")

def verify_evidence(query: str, retrieved_documents: List[Dict[str, Any]]) -> VerificationResult:
    if not retrieved_documents:
        return VerificationResult(
            supported=False, 
            confidence="LOW", 
            reason="No relevant regulatory sources were found within the similarity threshold."
        )

    context = "\n\n".join([f"Chunk: {doc['content']}" for doc in retrieved_documents])
    llm = get_llm()
    structured_llm = llm.with_structured_output(VerificationResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the provided Context contains sufficient factual evidence to answer the Question. "
                   "Return supported=True only if the answer is explicitly present in the context. "
                   "If the context is irrelevant or missing details, return supported=False."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt | structured_llm
    return chain.invoke({"context": context, "question": query})

# --- 4. Answer Generation ---
def generate_grounded_answer(query: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "I couldn't find sufficient evidence in my regulatory sources to answer this reliably."

    context_texts = []
    sources_used = set()
    for doc in evidence:
        src = doc['source']
        page = doc['page']
        sources_used.add((src, page))
        context_texts.append(f"[Source: {src}, Page: {page}]\n{doc['content']}")
        
    context_str = "\n\n".join(context_texts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Ayurveda Regulation Assistant. Answer strictly using the provided Context. "
                   "Do not hallucinate. Always include inline citations like [Source: X, Page: Y] when stating facts."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({"context": context_str, "question": query}).content
    
    # Append structured citations
    response += "\n\n### Sources Used:\n"
    for src, page in sources_used:
        response += f"- **{src}** (Page: {page})\n"
        
    return response

# --- 5. Checklist Generation (Optional extension) ---
def generate_compliance_checklist(product_type: str) -> str:
    # Example specific tool that retrieves and formats as a checklist
    docs = retrieve_regulations(f"licensing and labeling requirements for {product_type}")
    if not docs:
        return "No compliance data found for this product type."
    
    return generate_grounded_answer(
        f"Generate a strict step-by-step compliance checklist for {product_type} based on the context.", 
        docs
    )
