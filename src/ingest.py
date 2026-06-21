from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Loading PDF...")

loader = PyPDFLoader("data/ai_fundamentals.pdf")
docs = loader.load()

print(f"Pages loaded: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Chunks created: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector database...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_db"
)

print("Vector database created successfully!")
