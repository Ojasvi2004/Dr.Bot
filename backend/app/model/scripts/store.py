import os

os.environ["HF_HOME"] = "E:/hf_cache"
import faiss
from langchain_community.vectorstores import FAISS
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from pinecone import Pinecone,ServerlessSpec

from embed import _load_chunks

from train_Config import EMBEDDING_DIRECTORY,META_DATA_DIRECTORY,CHUNKS_DIRECTORY

from langchain.embeddings.base import Embeddings
import numpy as np
from dotenv import load_dotenv
load_dotenv()

class RealEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    
    
# def build_vector_db():
#     embeddings=np.load(EMBEDDING_DIRECTORY)
#     print(f"Loaded embeddings: {embeddings.shape}")
    
#     with open(META_DATA_DIRECTORY,'r',encoding="utf-8") as f:
#         meta_data=json.load(f)
#     print(f" Loaded metadata: {len(meta_data)} items")     
    
#     documents=[]
#     for item in meta_data:
#         text=item["chunk_text"]
#         source_file=item["source_file"]
#         chunnk_num=item["chunk_number"]
#         documents.append(Document(
#             page_content=text,
#              metadata={
#             "source_file": source_file,
#             "chunk_number": chunnk_num
#         }
#         ))
        
#     vector_dim=embeddings.shape[1]
#     index=faiss.IndexFlatIP(vector_dim)
#     index.add(embeddings)
#     real_embed = RealEmbeddings()
#     vector_store = FAISS(
#     index=index,
#     docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
# ,
#     index_to_docstore_id={i: str(i) for i in range(len(documents))},
#     embedding_function=real_embed
# )
    
    
#     print(" Vector DB Ready with documents")

#     return vector_store
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def build_pine_cone():
    index_name = "medical-rag-bge"
    embeddings=np.load(EMBEDDING_DIRECTORY)

    if not pc.has_index(index_name):
        pc.create_index(
                name=index_name,
                dimension=embeddings.shape[1], 
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index=pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        print("Index already populated. Skipping upsert.")
        return
    records=[]
    chunks,meta_data=_load_chunks(CHUNKS_DIRECTORY)
    
    for i,meta in enumerate(meta_data):
       records.append({
            "id": f"chunk-{i}",
            "values": embeddings[i].tolist(),  
             "metadata": {
            "text": meta_data[i]["chunk_text"],
            "source_file": meta_data[i]["source_file"],
            "chunk_number": meta_data[i]["chunk_number"]
        }
})
    index.upsert(records,batch_size=100)
    print(f"Upserted {len(records)} into Pinecone")
    
    



if __name__ == "__main__":
    
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"API Key found: {'Yes' if os.getenv('PINECONE_API_KEY') else 'No'}")
    build_pine_cone()