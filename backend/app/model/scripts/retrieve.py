from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

from store import build_vector_db

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["PWD"] = os.getcwd()

import sys

sys.stdout.reconfigure(encoding='utf-8')


llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

prompt = PromptTemplate(
    input_variables=["context", "question","patient"],
    template="""
You are a helpful medical assistant. Answer the question based ONLY on the context.

Patient Context:
{patient}

Medical Context:
{context}

Question:
{question}

Answer concisely:
"""
)


from langchain_core.runnables import RunnableLambda, RunnableSequence
def rag_chain_lcel(vector_store):

    retriever = vector_store.as_retriever(search_kwargs={"k": 30})

    def print_docs(docs):
        print("\n====== Retrieved Documents ======")
        for i, d in enumerate(docs, 1):
            print(f"\nDocument #{i}")
            print("Metadata:", d.metadata)
            # print("Content:\n", d.page_content)
            print("\n==============================")
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
        final_prompt = prompt.format(
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

    return full_pipeline

    





if __name__ == "__main__":
    vector_store = build_vector_db()
    rag_pipeline = rag_chain_lcel(vector_store)
    
    patient_summary = {
        "visits": [
            {"month": "Jan", "complaints": ["fatigue", "frequent urination"], "labs": {"HbA1c": 8.0}},
            {"month": "Apr", "complaints": ["blurry vision"], "medications": ["Metformin"]},
            {"month": "Jul", "labs": {"HbA1c": 7.2}, "vitals": {"BP": "140/90"}}
        ],
        "current_symptoms": ["tiredness sometimes", "mild headache in mornings"]
    }

    query = {
        "question": "Summerize this patient data and tell if it has any abnormal conditons.",
        "patient": patient_summary
    }

    answer = rag_pipeline.invoke(query)
    print("Answer:\n", answer)
