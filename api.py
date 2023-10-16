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

class Comments(BaseModel):
    texts: list[str]

MODEL_PATH = 'Model/Combined_K-Fold_NoEarlyStopping.pt'
TOKENIZER_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

@app.post("/predict/")
async def predict_sentiments(comments: Comments):
    results = {}
    for idx, text in enumerate(comments.texts):
        # Preprocess the input data
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=100,  # or whatever the MAX_LEN was set to during training
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Make predictions
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
        _, prediction = torch.max(logits, dim=1)
        sentiment_mapping = {0: 'negative', 1: 'positive'}
        predicted_label = sentiment_mapping[prediction.item()]

        results[idx] = {'text': text, 'sentiment': predicted_label}

    return results
