import os
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

GEMINI_API_KEY = os.getenv("AIzaSyCRNoHYfsfhDtJK9Mp_UB6b_Jrh4NCN7Ok")
genai.configure(api_key=GEMINI_API_KEY)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def build_vector_db(pdf_file):
    text = extract_text_from_pdf(pdf_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    embeddings = embedder.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


def search_chunks(query, index, chunks, top_k=5):
    q_embed = embedder.encode([query])
    distances, indices = index.search(q_embed, top_k)

    results = [chunks[i] for i in indices[0]]
    return results


def answer_query(query, index, chunks):
    context = "\n\n".join(search_chunks(query, index, chunks))

    model = genai.GenerativeModel("models")

    prompt = f"""
You are an expert assistant. Use ONLY the following context:

{context}

Question: {query}

Answer:
"""

    response = model.generate_content(prompt)
    return response.text
