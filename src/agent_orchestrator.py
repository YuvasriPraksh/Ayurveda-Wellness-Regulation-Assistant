import os
from typing import List, Dict, Any

from src.agent_tools import classify_query, retrieve_regulations, verify_evidence, generate_grounded_answer, decompose_query


def process_agentic_query(query: str) -> Dict[str, Any]:
    """Orchestrates the agentic RAG pipeline with support for multi‑step queries.

    Returns a state dict with:
    - query
    - intents (list of strings)
    - retrieved_documents (list of dicts)
    - evidence_status ("SUPPORTED" or "INSUFFICIENT")
    - confidence (str)
    - answer (str)
    - trace (list of step descriptions)
    """
    trace: List[str] = []
    trace.append("[INFO] Received user query")

    # Decompose possible compound queries
    subqueries = decompose_query(query)
    if len(subqueries) > 1:
        trace.append(f"[INFO] Decomposed into {len(subqueries)} sub‑queries")
    all_answers: List[str] = []
    all_retrieved: List[Dict[str, Any]] = []
    overall_supported = True
    overall_confidence = "high"

    for subq in subqueries:
        # 1. Intent classification per sub‑query
        classification = classify_query(subq)
        intents = classification.intents
        trace.append(f"[INFO] Sub‑query intents: {', '.join(intents)}")

        if "OUT_OF_SCOPE" in intents:
            trace.append("[WARN] Sub‑query out‑of‑scope – skipping")
            continue

        # 2. Retrieval
        retrieved = retrieve_regulations(subq, k=5)
        all_retrieved.extend(retrieved)
        trace.append(f"[INFO] Retrieved {len(retrieved)} docs for sub‑query")

        # 3. Evidence verification
        ev = verify_evidence(subq, retrieved)
        if not ev.supported:
            overall_supported = False
            overall_confidence = ev.confidence.lower()
            trace.append(f"🚫 Sub‑query insufficient evidence: {ev.reason}")
            continue
        # 4. Answer generation
        ans = generate_grounded_answer(subq, retrieved)
        all_answers.append(ans)
        trace.append("📝 Generated answer for sub‑query")

    # Aggregate results
    if not all_answers:
        answer = "I'm sorry, but the available regulatory sources do not cover this question."
        evidence_status = "INSUFFICIENT_EVIDENCE"
    else:
        answer = "\n\n".join(all_answers)
        evidence_status = "SUPPORTED" if overall_supported else "INSUFFICIENT_EVIDENCE"
    confidence = overall_confidence

    return {
        "query": query,
        "intents": classification.intents if subqueries else [],
        "retrieved_documents": all_retrieved,
        "evidence_status": evidence_status,
        "confidence": confidence,
        "answer": answer,
        "trace": trace,
    }
