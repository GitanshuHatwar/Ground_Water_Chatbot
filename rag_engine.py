import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter


# ---------------------- BUILD VECTOR DB ----------------------
def build_vector_db(pdf_folder="docs", db_path="vector_store"):
    documents = []

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(db_path)

    return vector_db


# ---------------------- QUERY ENGINE ----------------------
def answer_query(query, db_path="vector_store"):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    You are an expert assistant. Use ONLY the following context to answer.
    If context does not include the answer, say "Not found in documents."

    Context:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    return response.content
