import os
from dotenv import load_dotenv

load_dotenv()
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import openai
# -----------------------------------------------------------------
# 5. Multilingual Translation (uses OpenAI if API key is set)
# -----------------------------------------------------------------
def translate_text(text: str, target_language: str) -> str:
    """Translate `text` to `target_language` using OpenAI API if available.
    If `OPENAI_API_KEY` is not set, returns the original text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text
    openai.api_key = api_key
    try:
        # Use a more capable model if available; avoid hard‑coding gpt‑3.5‑turbo
        model_name = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a translation assistant. Translate the following text faithfully, preserving terminology. Do not add, omit, or modify content beyond translation."},
                {"role": "user", "content": f"Translate the following to {target_language}:\n\n{text}"}
            ],
            temperature=0,
        )
        translated = response["choices"][0]["message"]["content"].strip()
        return translated
    except Exception:
        return text

# Simple query decomposition for multi-step questions.
def decompose_query(query: str) -> List[str]:
    """Naively split a compound query into sub‑queries."""
    if ";" in query:
        parts = [p.strip() for p in query.split(";") if p.strip()]
        if len(parts) > 1:
            return parts
    if " and " in query.lower():
        parts = [p.strip() for p in query.split(" and ") if p.strip()]
        if len(parts) > 1:
            return parts
    return [query]

# Threshold for FAISS similarity (kept from previous implementation)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.1"))

# -----------------------------------------------------------------
# Helper: load_vector_store (imported lazily within retrieve function)
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# 1. Intent Classification (rule‑based, no external LLM calls)
# -----------------------------------------------------------------
class IntentClassificationResult(BaseModel):
    intents: List[str] = Field(description="List of identified intents: LICENSING, LABELING, IP, GENERAL_REGULATION, OUT_OF_SCOPE")
    reasoning: str = Field(description="Why these intents were chosen.")

def classify_query(query: str) -> IntentClassificationResult:
    """Simple keyword‑based intent detection.
    Returns a deterministic result and works without any API keys.
    """
    lower = query.lower()
    intents: List[str] = []
    if any(kw in lower for kw in ["licens", "license", "permit", "approval", "registration"]):
        intents.append("LICENSING")
    if any(kw in lower for kw in ["label", "labelling", "information", "packaging", "contents"]):
        intents.append("LABELING")
    if any(kw in lower for kw in ["ip", "patent", "intellectual property", "protection", "copyright"]):
        intents.append("IP")
    if any(kw in lower for kw in ["regulation", "require", "compliance", "standard", "rule"]):
        intents.append("GENERAL_REGULATION")
    if not intents:
        intents = ["OUT_OF_SCOPE"]
    return IntentClassificationResult(intents=intents, reasoning="Keyword based classification.")

# -----------------------------------------------------------------
# 2. Retrieval (unchanged – uses FAISS index)
# -----------------------------------------------------------------
def retrieve_regulations(query: str, k: int = 4) -> List[Dict[str, Any]]:
    from src.document_processor import load_vector_store
    vectorstore = load_vector_store()
    if not vectorstore:
        return []
    results = vectorstore.similarity_search_with_score(query, k=k)
    valid_chunks = []
    for doc, score in results:
        if score <= SIMILARITY_THRESHOLD:
            valid_chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Not available'),
                "section": doc.metadata.get('section', ''),
                "score": float(score)
            })
    return valid_chunks

# -----------------------------------------------------------------
# 3. Evidence Verification (deterministic heuristic)
# -----------------------------------------------------------------
class VerificationResult(BaseModel):
    supported: bool = Field(description="True if the retrieved documents contain sufficient evidence to answer the query, False otherwise.")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW confidence.")
    reason: str = Field(description="Reason why the evidence is sufficient or insufficient.")

def verify_evidence(query: str, retrieved_documents: List[Dict[str, Any]]) -> VerificationResult:
    if not retrieved_documents:
        return VerificationResult(supported=False, confidence="LOW", reason="No relevant regulatory sources were found within the similarity threshold.")
    # Simple heuristic: check if any key term from the query appears in the retrieved content.
    query_terms = set(query.lower().split())
    for doc in retrieved_documents:
        doc_terms = set(doc["text"].lower().split())
        if query_terms & doc_terms:
            return VerificationResult(supported=True, confidence="HIGH", reason="Relevant evidence found in retrieved documents.")
    return VerificationResult(supported=False, confidence="MEDIUM", reason="Documents retrieved but none directly address the query.")

# -----------------------------------------------------------------
# 4. Answer Generation (deterministic, no LLM)
# -----------------------------------------------------------------
def generate_grounded_answer(query: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "I couldn't find sufficient evidence in my regulatory sources to answer this reliably."
    answer_parts = []
    sources_used = set()
    for doc in evidence:
        src = doc["source"]
        page = doc.get("page", "Not available")
        sources_used.add((src, page))
        answer_parts.append(f"[Source: {src}, Page: {page}]\n{doc['text']}")
    answer_body = "\n\n".join(answer_parts)
    sources_section = "\n\n### Sources Used:\n"
    for src, page in sources_used:
        sources_section += f"- **{src}** (Page: {page})\n"
    return answer_body + sources_section

# -----------------------------------------------------------------
# 5. Checklist Generation (unchanged – re‑uses retrieve & answer functions)
# -----------------------------------------------------------------
def generate_compliance_checklist(product_type: str) -> str:
    """Generate a structured compliance checklist for the given product type.

    The function uses a predefined list of regulatory questions for each product type,
    retrieves evidence from the knowledge base, verifies it, and assembles a markdown
    checklist where each item includes:
        - Requirement (question)
        - Explanation (grounded answer if evidence is found)
        - Evidence status (SUPPORTED / INSUFFICIENT_EVIDENCE)
        - Source, page, and section when available
    If evidence is insufficient, the explanation states that it was not established.
    """
    # Define a small set of compliance questions per product type
    questions_map = {
        "ayurvedic medicine": [
            "What licences are required to manufacture an Ayurvedic medicine?",
            "What information must appear on the label of an Ayurvedic medicine?",
            "What GMP (Good Manufacturing Practice) requirements apply to Ayurvedic medicines?",
        ],
        "ayurvedic cosmetic": [
            "What licences are required to manufacture an Ayurvedic cosmetic product?",
            "What labelling requirements apply to Ayurvedic cosmetics?",
            "What safety and quality standards are mandated for Ayurvedic cosmetics?",
        ],
        "traditional formulation": [
            "What regulatory approvals are needed for a traditional Ayurvedic formulation?",
            "What post‑market surveillance obligations exist for traditional formulations?",
        ],
        "herbal wellness product": [
            "What licences are required for a herbal wellness product?",
            "What labelling information is mandatory for herbal wellness products?",
        ],
    }

    # Normalise product_type to match keys
    key = product_type.lower()
    questions = questions_map.get(key, [])
    if not questions:
        return "No compliance questions defined for this product type."

    checklist_items = []
    for q in questions:
        # Retrieve evidence for the question
        retrieved = retrieve_regulations(q, k=5)
        ev = verify_evidence(q, retrieved)
        if ev.supported:
            answer = generate_grounded_answer(q, retrieved)
            # Extract first source info for citation display
            src = retrieved[0].get('source', 'Unknown')
            page = retrieved[0].get('page', 'Not available')
            section = retrieved[0].get('section', '')
            citation = f"{src} – Page: {page}" + (f", Section: {section}" if section else "")
            item = f"- **Requirement**: {q}\n  **Explanation**: {answer}\n  **Evidence**: SUPPORTED\n  **Source**: {citation}\n"
        else:
            item = f"- **Requirement**: {q}\n  **Explanation**: Not established from available sources.\n  **Evidence**: INSUFFICIENT_EVIDENCE\n"
        checklist_items.append(item)
    # Compute a simple readiness percentage based on supported items
    supported_count = sum(1 for q in questions if verify_evidence(q, retrieve_regulations(q, k=5)).supported)
    readiness = int((supported_count / len(questions)) * 100) if questions else 0
    header = f"## Compliance Checklist for {product_type.title()}\nReadiness: {readiness}%\n\n"
    return header + "\n".join(checklist_items)


# Simple language detection (English/Tamil/Hindi)
def detect_language(text: str) -> str:
    """Return 'en', 'ta' (Tamil) or 'hi' (Hindi). Default to 'en' if unknown.
    Uses Unicode ranges for Tamil (U+0B80‑U+0BFF) and Hindi (Devanagari U+0900‑U+097F).
    """
    for ch in text:
        code = ord(ch)
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"
        if 0x0900 <= code <= 0x097F:
            return "hi"
    return "en"
