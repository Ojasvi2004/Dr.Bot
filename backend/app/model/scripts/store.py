import faiss
from langchain_community.vectorstores import FAISS
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore



from train_Config import EMBEDDING_DIRECTORY,META_DATA_DIRECTORY

from langchain.embeddings.base import Embeddings
import numpy as np

class RealEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    
    
def build_vector_db():
    embeddings=np.load(EMBEDDING_DIRECTORY)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    with open(META_DATA_DIRECTORY,'r',encoding="utf-8") as f:
        meta_data=json.load(f)
    print(f" Loaded metadata: {len(meta_data)} items")     
    
    documents=[]
    for item in meta_data:
        text=item["chunk_text"]
        source_file=item["source_file"]
        chunnk_num=item["chunk_number"]
        documents.append(Document(
            page_content=text,
             metadata={
            "source_file": source_file,
            "chunk_number": chunnk_num
        }
        ))
        
    vector_dim=embeddings.shape[1]
    index=faiss.IndexFlatIP(vector_dim)
    index.add(embeddings)
    real_embed = RealEmbeddings()
    vector_store = FAISS(
    index=index,
    docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
,
    index_to_docstore_id={i: str(i) for i in range(len(documents))},
    embedding_function=real_embed
)
    
    
    print(" Vector DB Ready with documents")

    return vector_store


if __name__ == "__main__":
    build_vector_db()