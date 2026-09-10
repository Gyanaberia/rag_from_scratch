import hashlib
import threading
import itertools
import sys
import time
import os
from sentence_transformers import SentenceTransformer #type: ignore
from chromadb import PersistentClient #type:ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

class VectorDB:
    db_path = None
    collection_name = None
    knowledge_base_path = None
    dbclient = None
    db_collection = None
    embedding_model = None

    def __init__(self, db_path=os.getenv("DB_PATH"), collection_name = os.getenv("COLLECTION_NAME"), knowledge_base_path=os.getenv("KNOWLEDGE_BASE_PATH")):
        self.db_path = db_path
        self.collection_name = collection_name
        self.knowledge_base_path = knowledge_base_path
        self.dbclient = PersistentClient(path=db_path)
        self.db_collection = self.dbclient.get_or_create_collection(collection_name)
        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

    #Calculate hash of text chunk
    def calculate_text_hash(self,text):
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    #chunk file function

    def read_file_by_chunks(self, file_path, chunk_size=1000):
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    #function for embedding creation
    def create_embeddings(self, text):
        embeddings = self.embedding_model.encode(text)
        return embeddings


    #function to read files in folder,chunk them and create embeddings and store them in a vector database
    def process_knowledge_base(self):
        stop_event = threading.Event()

        spinner_thread = threading.Thread(
            target=self.spinner,
            args=(stop_event,)
        )

        spinner_thread.start()
        #using a persistent client to store the embedding permanently in disk.
        #This way, only new or updated files needs to be processed.
        files = [
            filename
            for filename in os.listdir(self.knowledge_base_path)
            if filename.endswith(".txt")
        ]

        try:
            for file_index, filename in enumerate(files, start=1):
                if(filename.endswith('.txt')):
                    chunk_id = 1
                    for chunk in self.read_file_by_chunks(os.path.join(self.knowledge_base_path, filename),300):
                        # print(f"Processing {filename}, chunk {chunk_id}")
                        vector_id = filename + "_" + str(chunk_id)
                        # Using hash to check if the existing chunk has changed. 
                        # If it has changed or is new, we create a new embedding and store it in the vector database. 
                        # If it hasn't changed, we skip creating a new embedding to save time and resources.
                        # Not using the simpler db_collection.upsert() method, because it adds a item if id doesn't exist, 
                        # or update it as per given values.
                        # In both case we would have to provide the embedding, which is compute heavy. 
                        # Thus using the hash to check if the chunk is new or changed first.
                        current_hash = self.calculate_text_hash(chunk)
                        existing_vector = self.db_collection.get(ids=[vector_id],include=['metadatas','documents'])
                        #if it doesn't exist, save the embedding in db
                        if not existing_vector['metadatas']:
                            # print(f"Creating new embedding for {filename}, chunk {chunk_id}")
                            embedding = self.create_embeddings(chunk)
                            metadata = {"chunk_hash": current_hash}
                            self.db_collection.add(
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[metadata],
                                ids=[vector_id]
                            )
                        #if it doesn't match, update the embedding in db
                        elif existing_vector['metadatas'][0].get('chunk_hash')!=current_hash:
                            # print(f"Updating embedding for {filename}, chunk {chunk_id}")
                            embedding = self.create_embeddings(chunk)
                            self.db_collection.update(
                                ids=[vector_id],
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{"chunk_hash": current_hash}])
                        chunk_id += 1
        finally:
            stop_event.set()
            spinner_thread.join()

    def query_knowledge_base(self, query, top_k=3):
        #create embedding for the query
        query_embedding = self.create_embeddings(query)
        #query the vector database for the most relevant chunks
        results = self.db_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents']
        )
        return results.get('documents')[0]

    def spinner(self, stop_event):
        for char in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break

            sys.stdout.write(
                f"\rPlease wait... Processing knowledge base {char}"
            )
            sys.stdout.flush()

            time.sleep(0.1)

        sys.stdout.write(
            "\rKnowledge base processing completed\n")

