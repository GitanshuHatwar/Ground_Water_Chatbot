import streamlit as st
import json
from rag_engine import answer_query, build_vector_db

st.set_page_config(page_title="Groundwater RAG Assistant", layout="wide")

st.title("🌊 Groundwater Intelligence RAG Assistant")

# Sidebar - Build FAISS Index
st.sidebar.header("⚙️ System Setup")

if st.sidebar.button("Build Vector DB"):
    build_vector_db()
    st.sidebar.success("Vector DB built successfully!")

st.subheader("User Query")
query = st.text_input("Ask something about groundwater:")

st.subheader("Database JSON Input")
json_text = st.text_area("Paste JSON from database", height=250)

if st.button("Get Answer"):
    try:
        db_json = json.loads(json_text)
        result = answer_query(query, db_json)
        st.success("Result:")
        st.write(result)
    except Exception as e:
        st.error(f"Error: {e}")
