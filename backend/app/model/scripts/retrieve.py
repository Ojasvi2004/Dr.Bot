import os

os.environ["HF_HOME"] = "E:/hf_cache"
from pinecone import Pinecone,ServerlessSpec
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_pinecone import PineconeVectorStore
from store import RealEmbeddings

from transformers import AutoProcessor,AutoModelForImageTextToText
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()
os.environ["PWD"] = os.getcwd()

import sys

sys.stdout.reconfigure(encoding='utf-8')


import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     llm_int8_enable_fp32_cpu_offload=True
# )


# processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")


# model = AutoModelForImageTextToText.from_pretrained(
#     "google/medgemma-4b-it",
#     quantization_config=quant_config, 
#     device_map="auto",               
#     low_cpu_mem_usage=True,
#     token=os.getenv("HF_Token")
# )


def testing_data(path,index):
    with open(path,'r') as f:
        data=json.load(fp=f)
    return data[index]

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

from prompt_templates import prompt1
embeddings=RealEmbeddings()

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vector_store = PineconeVectorStore(
    index_name="medical-rag-bge", 
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)


from langchain_core.runnables import RunnableLambda, RunnableSequence

def rag_chain_lcel(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 30})

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
        
        
    # def MedGamma(x):
    #     medgamma_prompt_text=[{
    #         f"""
    #         Patient Context:{x["patient"]}
    #         Medical Context:{x["context"]}
    #         Query:{x["query"]}"""
    #     }]
        

    #     prompt_text = processor.apply_chat_template(
    #     medgamma_prompt_text, 
    #     add_generation_prompt=True, 
    #     tokenize=False
    # )
    #     inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)
        
    #     with torch.no_grad():
    #         outputs_id = model.generate(
    #         **inputs, 
    #         max_new_tokens=150,
    #         do_sample=True,
    #         temperature=0.7
    #     )
            
    #     decoded = processor.decode(outputs_id[0], skip_special_tokens=True)
    #     return decoded.replace(prompt_text, "").strip()
    
    def llm_with_patient(x):
        final_prompt = prompt1.format(
            context=x["context"],
            question=x["query"],
            patient=json.dumps(x.get("patient", {}), indent=2)
        )
        return llm.invoke(final_prompt)


    full_pipeline = RunnableSequence(
        
        knowledge_pipeline,
        
        RunnableLambda(llm_with_patient),
        
        
        StrOutputParser()
    )
    
    # full_pipeline2 = RunnableSequence(
        
    #     knowledge_pipeline,
        
    #     RunnableLambda(MedGamma),
        
    # )

    return full_pipeline

    





if __name__ == "__main__":
    gemini_pipeline = rag_chain_lcel(vector_store)
    
    
    query=testing_data("D:/ML/Dr.Bot/backend/app/model/scripts/script_testing/testing_queries.json",0)


    answer = gemini_pipeline.invoke(query)
    # answer2=medgamma_pipeline.invoke(query)
    print("Answer:/n", answer)
