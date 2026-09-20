import streamlit as st
from src.agent_orchestrator import process_agentic_query

# --- Page Configuration ---
st.set_page_config(
    page_title="Ayurveda Regulation Assistant",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Polish ---
st.markdown("""
<style>
    .main {
        background-color: #fcfcfc;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    h1 {
        color: #2e7d32;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3798/3798150.png", width=100)
    st.title("🌿 AYUSH Assistant")
    st.markdown("""
    Welcome to the **Agentic Ayurveda & Wellness Regulation Assistant**.
    
    This AI tool helps manufacturers, startups, and researchers navigate:
    - AYUSH licensing
    - Product labeling
    - Intellectual Property (IP)
    - Formulation regulations
    
    **Features:**
    - 📚 Agentic RAG Pipeline.
    - 🛡️ Strict fallback with Evidence Verification.
    - 🔍 Expandable Execution Trace.
    """)
    st.divider()
    st.caption("Powered by LangGraph & FAISS")

# --- Main Chat Interface ---
st.title("Ayurveda & Wellness Regulation Assistant")
st.markdown("Ask any questions related to Ayurvedic formulations, licensing, or IP.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Agentic Ayurveda Regulation Assistant. How can I help you today?"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there's an agent trace in history, display it inside an expander
        if "trace" in message and message["trace"]:
            with st.expander("🔍 Agent Activity Log"):
                for step in message["trace"]:
                    st.text(step)
                if "evidence_status" in message:
                    st.markdown(f"**Evidence Status**: {message['evidence_status']}  \n**Confidence**: {message['confidence']}")

# React to user input
if prompt := st.chat_input("E.g., What are the labeling requirements for Ayurvedic medicines?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Processing with Agent..."):
            try:
                # Call Agentic Pipeline
                state = process_agentic_query(prompt)
                
                # Display Answer
                st.markdown(state["answer"])
                
                # Display Trace
                with st.expander("🔍 Agent Activity Log", expanded=False):
                    for step in state["trace"]:
                        st.text(step)
                    st.markdown(f"**Evidence Status**: {state['evidence_status']}  \n**Confidence**: {state['confidence']}")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": state["answer"],
                    "trace": state["trace"],
                    "evidence_status": state["evidence_status"],
                    "confidence": state["confidence"]
                })
            except Exception as e:
                st.error(f"Configuration Error: {e}")
                st.info("Please make sure your .env file is set up correctly.")
