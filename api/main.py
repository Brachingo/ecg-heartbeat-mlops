import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import CLASS_NAMES
from predict import load_model, predict as classify_signal

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield
    _model = None


app = FastAPI(
    title="ECG Heartbeat Classifier",
    description="Classifies a single ECG heartbeat into 5 arrhythmia categories (MIT-BIH).",
    version="1.0.0",
    lifespan=lifespan,
)


class HeartbeatRequest(BaseModel):
    signal: list[float] = Field(..., description="187 normalised amplitude values of one heartbeat window")

    @field_validator("signal")
    @classmethod
    def check_length(cls, v):
        if len(v) != 187:
            raise ValueError(f"signal must have exactly 187 values, got {len(v)}")
        return v


class HeartbeatResponse(BaseModel):
    predicted_class: int
    label: str
    probabilities: dict[str, float]


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "ECG Heartbeat Classifier"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/classes", tags=["info"])
def get_classes():
    return CLASS_NAMES


@app.post("/predict", response_model=HeartbeatResponse, tags=["inference"])
def predict(request: HeartbeatRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = classify_signal(request.signal, model=_model)
    return result
