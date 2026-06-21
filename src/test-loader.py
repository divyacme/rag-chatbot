from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/ai_fundamentals.pdf")

docs = loader.load()

print("Number of pages:", len(docs))
print("\nFirst page preview:\n")
print(docs[0].page_content[:500])
