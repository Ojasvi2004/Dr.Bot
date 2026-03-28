from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
from fastapi.middleware.cors import CORSMiddleware  # Keep this import
from apis.ocr import ocr_main_api_function
from apis.rag import rag_main_api_function

app = FastAPI()

# ==============================================================================
# 👇 START: ADD THIS CORS MIDDLEWARE BLOCK
# This block tells your backend server that it's okay to accept requests
# from your frontend application running on a different address (e.g., localhost:5173).
# ==============================================================================

origins = [
    "http://localhost:5173",  # The address of your React frontend
    "http://127.0.0.1:5173",  # Also add this for consistency
    "http://localhost:3000",  # A common address for other JS frameworks
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allows specific origins
    allow_credentials=True,      # Allows cookies
    allow_methods=["*"],         # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],         # Allows all headers
)

# ==============================================================================
# 👆 END: END OF CORS MIDDLEWARE BLOCK
# ==============================================================================


class OCRRequest(BaseModel):
    image_url: str


@app.get("/")
def home():
    return {"success": True, "message": "Server is running"}


@app.post("/ocr")
def get_ocr_data(request: OCRRequest):
    try:
        result = ocr_main_api_function(request.image_url)
        json_result = json.loads(result)
        print(json_result)
        return {
            "success": True,
            "data": json_result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

class QueryRequest(BaseModel):
    question: str
    patient: Optional[Dict[str, Any]] = {}

@app.post("/askDrBot")
def ask_dr_bot(request: QueryRequest):
    try:
        result = rag_main_api_function(request.model_dump())
        print("Main functions output-", result)
        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"{str(e)},error in the main.py"
        }