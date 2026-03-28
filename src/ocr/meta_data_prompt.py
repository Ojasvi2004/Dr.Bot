from langchain_core.prompts import PromptTemplate

metadata_ask = PromptTemplate(
    input_variables=["extracted_text"],
    template="""
You are an expert medical data extraction system.

Your task is to extract structured laboratory test data from noisy OCR text and return it in strict JSON format.

IMPORTANT RULES:
1. Output ONLY valid JSON. No explanations.
2. Extract all lab test entries.
3. Normalize and clean OCR errors (e.g., "magidl" → "mg/dL", "gmndl" → "g/dL").
4. Identify abnormal flags:
   - "H" → High
   - "L" → Low
5. If any value is unclear, set it to null.
6. Maintain consistent structure.
7. Do NOT include markdown like ```json

Return JSON in this format:
{{
  "tests": [
    {{
      "test_name": "",
      "observed_value": "",
      "unit": "",
      "reference_range": "",
      "flag": ""
    }}
  ]
}}

OCR TEXT:
----------------
{extracted_text}
----------------

Now extract the structured data.
"""
)