from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

query = input("Ask a question: ")

results = db.similarity_search(query, k=3)

print("\nTop Retrieved Chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content[:500])
