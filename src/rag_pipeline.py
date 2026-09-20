import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.document_processor import load_vector_store

load_dotenv()

# Threshold for distance (FAISS L2 distance). 
# Lower distance means more similar. Adjust based on your embedding model.
# all-MiniLM-L6-v2 outputs normalized embeddings, L2 distance is usually between 0 and 2.
# 1.2 is a reasonable starting threshold for relevance.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.1"))

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError("Google API Key not set. Please update the .env file.")
    
    # Use gemini-1.5-flash for fast, high-quality responses
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def answer_query(query: str):
    """
    Retrieves context for the query, applies the NOT-IN-MY-SOURCES fallback,
    and returns the grounded answer with citations.
    """
    vectorstore = load_vector_store()
    if not vectorstore:
        return "System error: Document knowledge base is empty or could not be loaded."

    # Perform similarity search with scores
    # FAISS returns (document, score). For L2 distance, lower score is better.
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    # Filter by threshold
    valid_results = [res for res in results if res[1] <= SIMILARITY_THRESHOLD]
    
    if not valid_results:
        # MANDATORY FEATURE: NOT-IN-MY-SOURCES FALLBACK
        return "I'm sorry, but the available regulatory sources do not cover this question. As an Ayurveda & Wellness Regulation Assistant, I am strictly instructed not to provide answers outside of my retrieved documents to ensure accuracy and compliance."
    
    # Prepare context and sources
    context_texts = []
    sources = set()
    for doc, score in valid_results:
        context_texts.append(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}")
        sources.add(doc.metadata.get('source', 'Unknown'))
        
    context_str = "\n\n".join(context_texts)
    
    # Strict prompt to prevent hallucination
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are the Ayurveda & Wellness Regulation Assistant. "
                   "Your primary role is to answer questions based strictly on the provided Context. "
                   "If the Context does not contain sufficient information to confidently answer the question, "
                   "you MUST state that the sources do not cover the question. "
                   "Do not hallucinate or use outside knowledge. "
                   "Include source citations in your answer when providing facts."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    llm = get_llm()
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"context": context_str, "question": query})
        
        # Append source list for transparency
        final_answer = response.content + "\n\n### References:\n- " + "\n- ".join(sources)
        return final_answer
    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"
