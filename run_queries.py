import json
from src.agent_orchestrator import process_agentic_query

queries = [
    "What licence is required for an Ayurvedic medicine?",
    "What information should appear on an Ayurvedic medicine label?",
    "What licence is required and what information should appear on the label?",
    "What is the capital of France?"
]
for q in queries:
    state = process_agentic_query(q)
    print(json.dumps(state, indent=2, ensure_ascii=True))
    print("---")
