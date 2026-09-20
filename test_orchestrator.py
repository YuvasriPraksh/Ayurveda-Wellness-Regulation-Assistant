import os, sys
sys.path.append('.')
# Ensure the console can handle UTF‑8 (Windows default is cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
from src.agent_orchestrator import process_agentic_query
import json
state = process_agentic_query('What licence is required to manufacture an Ayurvedic medicine?')
print(json.dumps(state, indent=2, ensure_ascii=False))
