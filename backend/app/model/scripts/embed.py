from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json
from tqdm import tqdm

from train_Config import EMBEDDING_DIRECTORY,META_DATA_DIRECTORY,CHUNKS_DIRECTORY

model=SentenceTransformer("BAAI/bge-base-en-v1.5")


def _load_chunks(directory):
    chunks=[]
    metadata=[]
    files=[f for f in os.listdir(directory) if f.lower().endswith(".json")]
    if not files:
        print("Json not found")
        return chunks,metadata
    else:
        for fileName in tqdm(files,desc="Loading Chunks"):
            filePath=os.path.join(directory,fileName)
            
            with open(filePath,'r',encoding="utf-8") as f:
                data=json.load(f)
                
            for i,chunk in enumerate(data):
                chunks.append(chunk)
                metadata.append({
                    "source_file":fileName,
                    "chunk_number":i,
                    "chunk_text":chunk
                })
        print(f"[INFO] Total chunks loaded: {len(chunks)}")
    return chunks, metadata


def embed_chunks(directory):
    chunks,metadata=_load_chunks(directory)
    if len(chunks)==0:
        print("No chunks")
        return
    print(f"Sending {len(chunks)} for embedding...")
    
    embeddings= model.encode(
        chunks,
        batch_size=24,
        show_progress_bar=True,
        normalize_embeddings=True)
    
    np.save(EMBEDDING_DIRECTORY,embeddings)
    print(f" Embeddings saved to {EMBEDDING_DIRECTORY}")
    
    with open(META_DATA_DIRECTORY,"w",encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)
    
    print(f"Metadata saved to {META_DATA_DIRECTORY}")
    print(" Embedding process completed.")
    
if __name__=="__main__":
    embed_chunks(CHUNKS_DIRECTORY)
    
    