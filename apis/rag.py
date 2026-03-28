from pydantic import BaseModel
from typing import Optional,Dict,Any
import requests
from src.rag.components.retrieve import rag_chain_lcel,vector_store



class QueryRequest(BaseModel):
    question: str
    patient: Optional[Dict[str, Any]] = {}
    
gemini_pipeline = rag_chain_lcel(vector_store)
def rag_main_api_function(query: dict):
    try:
        # Support both top-level 'question' or nested under 'query'
        if "question" in query:
            q_text = query["question"]
            patient_data = query.get("patient", {})
        elif "query" in query and "question" in query["query"]:
            q_text = query["query"]["question"]
            patient_data = query["query"].get("patient", {})
        else:
            raise ValueError("Missing 'question' in payload")

        response = gemini_pipeline.invoke({
            "question": q_text,
            "patient": patient_data
        })
        print("API pipeline response-", response)

        return {
            "success": True,
            "result": response
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"{str(e)}, error in rag.py"
        }


