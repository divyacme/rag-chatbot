import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.title("📚 PDF RAG Chatbot")

# API Key check
st.write("API key found:", bool(os.getenv("GOOGLE_API_KEY")))

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

vector_db = None

if uploaded_file:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(pages)

    # Create vector DB (in-memory)
    vector_db = Chroma.from_documents(
        docs,
        embedding_model
    )

    st.success("PDF processed successfully!")

# Query section
query = st.text_input("Ask a question:")

if query and vector_db:

    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.write(f"Chunk {i}")
            st.write(doc.page_content[:500])