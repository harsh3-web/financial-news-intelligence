import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.data_utils import normalize_numbers

LABELS = ["negative", "neutral", "positive"]

class SentimentModel:
    def __init__(self, model_dir):
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir).eval()

    def predict(self, texts):
        clean = [normalize_numbers(t.strip()) for t in texts]          # same cleaning as training
        enc = self.tokenizer(clean, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = self.model(**enc).logits.softmax(dim=-1).numpy()
        return [{"label": LABELS[p.argmax()],
                 "confidence": round(float(p.max()), 4),
                 "probabilities": {l: round(float(v), 4) for l, v in zip(LABELS, p)}}
                for p in probs]