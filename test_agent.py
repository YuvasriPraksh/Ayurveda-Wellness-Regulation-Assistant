from src.agent_orchestrator import process_agentic_query

queries = [
    "What are the licensing requirements for Ayurvedic medicines?", # Licensing
    "What information must appear on the label of an Ayurvedic product?", # Labeling
    "Can I patent a traditional Ayurvedic formulation from Charaka Samhita?", # IP
    "What licence is required and what information must appear on the label?", # Multi-step
    "Who won the Cricket World Cup in 2011?", # Unrelated / Out of Scope
    "What is the permissible limit of Uranium in Ayurvedic medicine?" # Insufficient evidence
]

for q in queries:
    print(f"\n--- QUERY: {q} ---")
    state = process_agentic_query(q)
    print(f"Intents: {state['intents']}")
    print(f"Evidence Status: {state['evidence_status']} (Confidence: {state['confidence']})")
    print("Trace:")
    for t in state['trace']:
        print(f"  {t}")
    print(f"\nAnswer:\n{state['answer']}")
    print("-" * 50)
