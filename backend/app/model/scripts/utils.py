import os
import json
from transformers import AutoTokenizer
import pandas
import fitz

def load_json_dataset(file_path):
    data=[]
    with open(file_path, 'r' ,encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            data.append(item)
        return data

def load_text_file(file_path,chunk_size=512):
    data=[]
    with open(file_path,'r',encoding='utf-8') as f:
        text=f.read()
        for i in range(0,len(text),chunk_size):
            chunk=text[i:i+chunk_size]
            data.append(chunk)
    return data

def load_pdf(file_path):
    text=[]
    file=fitz.open(file_path)
    for page in file:
        text_data=page.get_text()
        text.append(text_data)
    return text_data

def read_doctorchat_csv(file_path):
    df=pandas.read_csv(file_path)
    return df[df['input','output']].to_dict(orient='records')
        
        

def save_checkpoints(model,save_dir,step=None):
    os.makedirs(save_dir,exist_ok=True)
    if step:
        path = os.path.join(save_dir, f"checkpoint_step_{step}")
    else:
        path = save_dir
    model.save_pretrained(path)
    print(f"Checkpoint saved at: {path}")
    
    
def get_tokenizer(model_path="meta-llama/Meta-Llama-3-8B"):
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"
    return tokenizer

