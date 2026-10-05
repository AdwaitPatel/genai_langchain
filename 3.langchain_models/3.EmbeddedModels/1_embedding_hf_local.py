from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

text = "Delhi is the capital of india"

documents = [
    "Delhi is the capital of india",
    "Kolkata is capital of west bengal",
    "Vada pav is food capital of mumbai"
]

# vector = embedding.embed_query(text)
vector = embedding.embed_documents(documents)

print(str(vector))

