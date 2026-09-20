from src.agent_tools import (
    classify_query, 
    retrieve_regulations, 
    verify_evidence, 
    generate_grounded_answer
)

def process_agentic_query(query: str) -> dict:
    """
    Orchestrates the agentic RAG pipeline.
    Maintains a structured execution state trace.
    """
    state = {
        "query": query,
        "intents": [],
        "retrieved_documents": [],
        "evidence_status": "PENDING",
        "confidence": "NONE",
        "answer": "",
        "trace": []
    }

    try:
        # Step 1: Classification
        state["trace"].append("🧠 Understanding query...")
        classification = classify_query(query)
        state["intents"] = classification.intents
        state["trace"].append(f"✓ Classified intent(s): {', '.join(classification.intents)}")

        # Check for Out of Scope
        if "OUT_OF_SCOPE" in state["intents"] and len(state["intents"]) == 1:
            state["evidence_status"] = "INSUFFICIENT"
            state["confidence"] = "LOW"
            state["answer"] = "⚠️ I couldn't find sufficient evidence in my regulatory sources to answer this reliably."
            state["trace"].append("❌ Query is outside the scope of Ayurveda regulations.")
            return state

        # Step 2: Retrieval
        state["trace"].append("🔍 Searching regulatory sources...")
        retrieved_docs = retrieve_regulations(query)
        state["retrieved_documents"] = retrieved_docs
        state["trace"].append(f"✓ Retrieved {len(retrieved_docs)} relevant chunks.")

        # Step 3: Verification
        state["trace"].append("🛡️ Verifying evidence...")
        verification = verify_evidence(query, retrieved_docs)
        state["evidence_status"] = "SUFFICIENT" if verification.supported else "INSUFFICIENT"
        state["confidence"] = verification.confidence
        state["trace"].append(f"✓ Evidence status: {state['evidence_status']} (Confidence: {state['confidence']})")
        state["trace"].append(f"  Reason: {verification.reason}")

        # Step 4: Grounding & Generation
        if not verification.supported:
            state["answer"] = "⚠️ I couldn't find sufficient evidence in my regulatory sources to answer this reliably."
            state["trace"].append("❌ Generation blocked: Insufficient verified evidence.")
        else:
            state["trace"].append("✍️ Generating grounded answer...")
            state["answer"] = generate_grounded_answer(query, retrieved_docs)
            state["trace"].append("✓ Answer generated successfully.")

        return state

    except Exception as e:
        state["answer"] = f"An internal error occurred: {str(e)}"
        state["trace"].append(f"❌ Error: {str(e)}")
        return state
