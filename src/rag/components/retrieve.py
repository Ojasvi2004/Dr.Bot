import os

os.environ["HF_HOME"] = "E:/hf_cache"
from pinecone import Pinecone,ServerlessSpec
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_pinecone import PineconeVectorStore
from src.rag.components.store import RealEmbeddings

from transformers import AutoProcessor,AutoModelForImageTextToText
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()
os.environ["PWD"] = os.getcwd()

import sys

sys.stdout.reconfigure(encoding='utf-8')


import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig



def testing_data(path,index):
    with open(path,'r') as f:
        data=json.load(fp=f)
    return data[index]

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

from src.rag.components.prompt_templates import prompt1
embeddings=RealEmbeddings()

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_store = PineconeVectorStore(
    index_name="medical-rag-bge", 
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)


from langchain_core.runnables import RunnableLambda, RunnableSequence

def rag_chain_lcel(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    def print_docs(docs):
        # print("/n====== Retrieved Documents ======")
        # for i, d in enumerate(docs, 1):
        #     print(f"/nDocument #{i}")
        #     print("Metadata:", d.metadata)
        #     # print("Content:/n", d.page_content)
        #     print("/n==============================")
        return docs


    knowledge_pipeline = RunnableSequence(
        RunnableLambda(lambda x: {"query": x["question"],
                                  "patient": x.get("patient", {})}),
        
        RunnableLambda(lambda x: {
            "query": x["query"],
            "docs": retriever.invoke(x["query"]),
            "patient": x["patient"]
        }),
        
        RunnableLambda(lambda x: {
            "query": x["query"],
            "docs": print_docs(x["docs"]),
            "patient": x["patient"]
        }),
        
        RunnableLambda(lambda x: {
            "query": x["query"],
            "context": "\n\n".join(d.page_content for d in x["docs"]),
            "patient": x["patient"]
        })
    )
        
        
  
    def llm_with_patient(x):
        final_prompt = prompt1.format(
            context=x["context"],
            question=x["query"],
            patient=json.dumps(x.get("patient", {}), indent=2)
        )
        raw_ouput=llm.invoke(final_prompt).content
        
        try:
            return json.loads(raw_ouput)
        except:
            return {
                "error":"Invalid  JSON frm LLM",
                "raw_ouput":raw_ouput
            }



    full_pipeline = RunnableSequence(
        
        knowledge_pipeline,
        
        RunnableLambda(llm_with_patient),
        
    )
    


    return full_pipeline

    





if __name__ == "__main__":
    gemini_pipeline = rag_chain_lcel(vector_store)
    
    
    query=testing_data("D:/ML/Dr.Bot/backend/app/model/scripts/script_testing/testing_queries.json",0)


    answer = gemini_pipeline.invoke(query)

    print("Answer:/n", answer)
