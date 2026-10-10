import sys, time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.predict import SentimentModel

MODEL_VERSION = "finbert-v1"
state = {}

@asynccontextmanager
async def lifespan(app):
    state["model"] = SentimentModel(ROOT / "models" / "finbert")     # load ONCE at start-up
    yield

app = FastAPI(title="Financial News Sentiment API", version="1.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32,
                             examples=[["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn ."]])

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    results = state["model"].predict(req.texts)
    return {"model_version": MODEL_VERSION,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": results}