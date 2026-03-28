from langchain_core.prompts import (PromptTemplate,
                                    ChatPromptTemplate,
                                    ChatMessagePromptTemplate)
from langchain_core.output_parsers import (StrOutputParser)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (RunnableParallel,
                                      RunnableMap,
                                      Runnable,
                                      RunnableLambda,
                                      RunnableSequence)
from .model import OCRReports


from dotenv import load_dotenv
import sys
load_dotenv()
import os

from .meta_data_prompt import metadata_ask

llm=ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

def ocr_chain(path):
    ocr_object=OCRReports(path)
    extracted_text=ocr_object.run()
    print(extracted_text)
    return extracted_text
    


def llm_for_ocr(x):
    final_prompt=metadata_ask.invoke({
        "extracted_text":x
    })
    return llm.invoke(final_prompt)


ocr_main_pipeline=RunnableSequence(
    RunnableLambda(ocr_chain),
    RunnableLambda(llm_for_ocr),
    StrOutputParser()
)

if __name__=="__main__":
    result=ocr_main_pipeline.invoke('src/ocr/test/BLR-0425-PA-0040652_LAB MERG_27-04-2025_1239-18_PM@E.pdf_page_7.png')
    print(result)
    
    
    
    


