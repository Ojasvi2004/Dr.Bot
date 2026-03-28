from pydantic import BaseModel
from typing import Optional
import requests
from PIL import Image
from io import BytesIO
from src.ocr.ocr_main import ocr_main_pipeline

class QueryRequest(BaseModel):
    query:str
    image_url:Optional[str]=None
    


def load_image_from_url(url:str):
    try:
        response=requests.get(url)
        image=Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        return {"error": str(e)}


import tempfile

def ocr_main_api_function(url: str):
    try:
        image = load_image_from_url(url)

  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

      
        result = ocr_main_pipeline.invoke(temp_path)

        return result

    except Exception as e:
        return {"error": str(e)}
    
    

    