from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records like hitting most centruies.",
    "Rohit Sharma is known for his elegant batting and record-breaking runs.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# query = "tell me about virat kohli" 
# query = "tell me about bumrah" 
# query = "who is known as the god of cricket" 
query = "who has scored maximum centuries" 

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("query : ", query)
print("response : ", documents[index])
print("similarity score is : ", score)






