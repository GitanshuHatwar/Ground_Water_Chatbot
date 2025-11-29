import streamlit as st
from rag_engine import build_vector_db, answer_query

st.set_page_config(page_title="RAG Gemini App", layout="wide")

st.title("📘 RAG Model - Jal_Sathi Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    if st.button("Build Vector Database"):
        with st.spinner("Processing..."):
            index, chunks = build_vector_db(uploaded_file)
            st.session_state.index = index
            st.session_state.chunks = chunks
        st.success("Vector DB ready!")

query = st.text_input("Ask a question based on the PDF")

if query:
    if "index" not in st.session_state:
        st.error("Please upload a PDF and build the vector DB first.")
    else:
        with st.spinner("Generating answer using Gemini..."):
            answer = answer_query(
                query,
                st.session_state.index,
                st.session_state.chunks
            )
        st.write("### 📌 Answer:")
        st.write(answer)
