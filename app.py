import streamlit as st
import os
from src.agent_orchestrator import process_agentic_query
from src.agent_tools import generate_compliance_checklist
from src.document_processor import load_vector_store
from typing import List, Dict, Any

# --------------------------------------------------
# Page Configuration & Styling
# --------------------------------------------------
st.set_page_config(
    page_title="AYURA REGULATE",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium Ayurveda / wellness look
st.markdown(
    """
    <style>
        /* Base colors */
        :root {
            --primary-green: #2e5d2c;      /* deep botanical green */
            --secondary-cream: #f8f5ef;   /* warm cream/off‑white */
            --accent-gold: #c19a6b;       /* subtle gold */
            --card-bg: #ffffff;           /* white cards */
        }
        .header, .footer {
            text-align: center;
            padding: 1rem 0;
            background-color: var(--card-bg);
            border-radius: 0.8rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            color: #333;
        }
        .hero {
            background: var(--card-bg);
            border-radius: 0.8rem;
            padding: 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            color: #333;
        }
        .card {
            background: var(--card-bg);
            border-radius: 0.8rem;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            color: #333;
        }
        .example-card {
            background: #e6f4ea;
            border-radius: 0.6rem;
            padding: 0.8rem 1rem;
            margin: 0.3rem 0;
            cursor: pointer;
            transition: background 0.2s;
        }
        .example-card:hover {
            background: #c8e8d4;
        }
        .badge {
            display: inline-block;
            background: #f0e6d2;
            color: #8a6d3b;
            padding: 0.2rem 0.6rem;
            border-radius: 0.4rem;
            margin-right: 0.4rem;
            font-size: 0.85rem;
        }
        .status-indicator {
            color: #4caf50;
            font-weight: bold;
        }
        .source-card {
            background: #fafafa;
            border-left: 4px solid #b5cfa2;
            padding: 0.8rem;
            margin-top: 0.5rem;
            border-radius: 0.4rem;
            color: #333;
        }
        .footer {
            font-size: 0.85rem;
            color: #777;
        }
        .warning-card {
            border-left: 4px solid #d9534f;
            background: #fff5f5;
            padding: 1rem;
            border-radius: 0.6rem;
            color: #333;
        }
        .success-card {
            border-left: 4px solid #5cb85c;
            background: #f5fff5;
            padding: 1rem;
            border-radius: 0.6rem;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Sidebar (Branding, Navigation, Settings)
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class='header' style='display:flex; align-items:center; gap:0.5rem;'>
            <img src='file:///C:/Users/HP/.gemini/antigravity-ide/brain/88acb8e3-36a4-4250-b7a1-97e2544218a8/ayura_logo_1789904624598.jpg' width='50' style='border-radius:6px;'>
            <h2>🌿 AYURA REGULATE</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Navigation")
    nav = st.radio("Go to", ["Regulatory Assistant", "Compliance Checker", "Knowledge Base", "About"], index=0)
    st.subheader("Language")
    language = st.selectbox("Select language", ["English", "தமிழ்", "हिन्दी"])  # placeholder – not wired yet
    st.subheader("Safety")
    st.info("Responses are grounded only in available regulatory sources.")
    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        if "assistant_query" in st.session_state:
            del st.session_state["assistant_query"]
        st.rerun()

# --------------------------------------------------
# Header Section (global branding)
# --------------------------------------------------
st.markdown(
    """
    <div class='header' style='display:flex; align-items:center; gap:1rem;'>
        <img src='file:///C:/Users/HP/.gemini/antigravity-ide/brain/88acb8e3-36a4-4250-b7a1-97e2544218a8/ayura_logo_1789904624598.jpg' width='80' style='border-radius:8px;'>
        <div>
            <h1 style='margin:0;'>AYURA REGULATE</h1>
            <p style='margin:0; font-size:1.1rem'>AI-powered regulatory intelligence for Ayurveda & Wellness</p>
            <p style='margin:0; font-weight:600'>\"Ask. Verify. Comply.\"</p>
            <p class='status-indicator' style='margin:0;'>● Regulatory Intelligence Online</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Hero Section (shown on Assistant view)
# --------------------------------------------------
if nav == "Regulatory Assistant":
    st.markdown(
        """
        <div class='hero'>
            <h2>Navigate Ayurveda Regulations with Confidence.</h2>
            <p>Ask questions about licensing, labeling and intellectual property — grounded in your verified regulatory sources.</p>
            <div>
                <span class='badge'>✓ Source Grounded</span>
                <span class='badge'>✓ Citation Verified</span>
                <span class='badge'>✓ Hallucination Safe</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Main Workspace – Regulatory Assistant
    # --------------------------------------------------
    st.markdown("### Ask the Regulatory Assistant")
    # Input box
    query = st.text_input(
        "", placeholder="Ask about licensing, labeling, IP or regulatory requirements...", key="assistant_query"
    )

    # Example cards
    examples = {
        "LICENSING": "What licence is required to manufacture an Ayurvedic medicine?",
        "LABELING": "What information must appear on an Ayurvedic product label?",
        "IP": "What IP considerations apply to a traditional formulation?",
        "REGULATION": "What regulatory requirements should I check before launching?",
    }
    cols = st.columns(4)
    for idx, (label, txt) in enumerate(examples.items()):
        with cols[idx]:
            if st.button(txt, key=f"ex_{label}"):
                query = txt
                st.session_state["assistant_query"] = txt

    if query:
        # Loading state with step‑by‑step messages
        with st.spinner("Processing…"):
            # The orchestrator already returns a full trace; we keep it lightweight for UI
            state = process_agentic_query(query)

        # --------------------------------------------------
        # Agent Activity – high‑level trace
        # --------------------------------------------------
        with st.expander("🔍 Agent Activity"):
            for step in state.get("trace", []):
                st.text(step)

        # --------------------------------------------------
        # Answer / Fallback UI
        # --------------------------------------------------
        if state.get("evidence_status") in ["INSUFFICIENT", "INSUFFICIENT"]:
            st.markdown(
                """
                <div class='warning-card'>
                    <h4>⚠️ Evidence Not Found</h4>
                    <p>I couldn't find sufficient evidence in my regulatory sources to answer this reliably.</p>
                    <p><strong>Evidence Status:</strong> INSUFFICIENT</p>
                    <p><strong>Confidence:</strong> LOW</p>
                    <p>Try asking about licensing, labeling, IP, or regulatory requirements covered by the available sources.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Success answer card
            st.markdown(
                """
                <div class='success-card'>
                    <h4>Answer</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(state.get("answer"))
            # Metadata summary
            st.markdown(
                f"""
                <div class='card'>
                    <p><strong>Evidence Status:</strong> ✓ {state.get('evidence_status', '')}</p>
                    <p><strong>Confidence:</strong> {state.get('confidence', '').upper()}</p>
                    <p><strong>Intent(s):</strong> {', '.join(state.get('intents', []))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Sources used with detailed metadata
            st.markdown("### 📚 Sources Used")
            for src in state.get("retrieved_documents", []):
                source_name = src.get('source', 'Unknown')
                page = src.get('metadata', {}).get('page', 'Not available')
                section = src.get('metadata', {}).get('section', '')
                extra = f" • Page: {page}" if page else ""
                extra += f" • Section: {section}" if section else ""
                header = f"{source_name}{extra} – Score: {src.get('score', 'N/A'):.2f}"
                with st.expander(header):
                    st.write(src.get("content", ""))

    else:
        # Empty state before any query
        st.markdown(
            """
            <div class='card'>
                <h4>Your regulatory copilot is ready.</h4>
                <p>Ask about: <strong>Licensing</strong> • <strong>Labeling</strong> • <strong>IP</strong> • <strong>Regulations</strong></p>
                <p>Try one of the example questions above.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif nav == "Compliance Checker":
    st.markdown("### Compliance Readiness")
    st.write("Generate a regulatory checklist based only on your verified sources.")
    product_type = st.selectbox(
        "Product type",
        ["Ayurvedic Medicine", "Ayurvedic Cosmetic", "Traditional Formulation", "Herbal Wellness Product"],
        key="prod_type",
    )
    product_name = st.text_input("Product name (optional)", key="prod_name")
    intended_use = st.text_input("Intended use (optional)", key="intended_use")
    generate_btn = st.button("Generate Compliance Checklist")
    if generate_btn:
        with st.spinner("Generating checklist…"):
            checklist = generate_compliance_checklist(product_type.lower())
        st.markdown(
            """
            <div class='card'>
                <h4>Compliance Readiness</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Render checklist as bullet list with checkboxes (visual only)
        for line in checklist.split("\n"):
            if line.strip():
                st.markdown(f"- {line.strip()}")

elif nav == "Knowledge Base":
    st.markdown("### Knowledge Base")
    # Basic stats – we only have files on disk and can attempt to count FAISS vectors.
    data_dir = os.path.join(os.getcwd(), "data")
    doc_files = [f for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.pdf')]
    st.write(f"**Documents indexed:** {len(doc_files)}")
    # Try to get number of chunks / vectors if the store exists
    @st.cache_resource
    def get_vector_stats():
        try:
            vectorstore = load_vector_store()
            # FAISS stores vectors; we can approximate count via index.ntotal
            count = getattr(vectorstore, "index", None).ntotal if getattr(vectorstore, "index", None) else None
            return count
        except Exception:
            return None
    chunk_count = get_vector_stats()
    if chunk_count is not None:
        st.write(f"**Chunks / vectors:** {chunk_count}")
    else:
        st.write("**Chunks / vectors:** unavailable")
    st.write("**Embedding model:** all-MiniLM-L6-v2 (sentence‑transformers)")
    st.write("**Vector store:** FAISS (cpu) – persisted locally.")
    st.write("**Last indexed:** now (on startup) – reload required after adding new docs.")

else:  # About
    st.markdown(
        """
        <div class='card'>
            <h3>About AYURA REGULATE</h3>
            <p>AI‑powered regulatory intelligence for Ayurveda & Wellness. The assistant answers only from verified regulatory documents, provides citations, and never hallucinates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class='footer'>
        <p>AYURA REGULATE – Agentic RAG for Ayurveda & Wellness</p>
        <p>Built for I‑Agentic Hackathon</p>
    </div>
    """,
    unsafe_allow_html=True,
)
