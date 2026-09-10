##to check if db is persistent

from chromadb import PersistentClient #type:ignore
import os
from vector_db import VectorDB
client = PersistentClient(path=os.getenv("DB_PATH"))
print(client.get_collection(name=os.getenv('COLLECTION_NAME')).count())
print(client.get_collection(name=os.getenv('COLLECTION_NAME')).peek())

db = VectorDB()
input_query = input("Enter your query: ")
top_k = input("Enter the number of top relevant chunks to retrieve (default is 1): ")
if not top_k.isdigit():
    top_k = 1
else:
    top_k = int(top_k)
results = db.query_knowledge_base(input_query, top_k)
print("Most relevant chunks from the knowledge base:")
for i, chunk in enumerate(results):
    print(f"Chunk {i+1}: {chunk}")