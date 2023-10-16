from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import thai2transformers
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SEFR_SPLIT_TOKEN
)

app = FastAPI()

# Define the input data model
class Comments(BaseModel):
    texts: list[str]

# Load Model and Tokenizer
MODEL_PATH = 'D:/Study/year3_1/AI/Project/API/Model/Combined_K-Fold_NoEarlyStopping.pt'
TOKENIZER_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

@app.post("/predict/")
async def predict_sentiments(comments: Comments):
    
    texts = comments.texts
    encodings = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=100, 
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
    _, predictions = torch.max(logits, dim=1)
    sentiment_mapping = {0: 'negative', 1: 'positive'}
    predicted_labels = [sentiment_mapping[pred.item()] for pred in predictions]

    results = dict(zip(texts, predicted_labels))
    return results