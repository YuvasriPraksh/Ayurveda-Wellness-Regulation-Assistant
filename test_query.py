import json
from src.agent_orchestrator import process_agentic_query

def test(query):
    state = process_agentic_query(query)
    print(json.dumps(state, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    test('What licence is required to manufacture an Ayurvedic medicine?')
    test('Who won the cricket world cup in 2011?')
