"""Local-only HTTP interface. Load only the trusted model shipped or trained by you."""
from functools import lru_cache
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from roadlab.core import ROOT, FEATURES, validate_record

app = FastAPI(title="RoadLab", version="1.0.0", description="Educational retrospective severity experiment; not a safety decision tool.")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@lru_cache
def resources():
    path = ROOT / "artifacts"
    return joblib.load(path / "model.joblib"), json.loads((path / "schema.json").read_text()), json.loads((path / "metrics.json").read_text())


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict] = Field(min_length=1, max_length=100)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
def health():
    try:
        resources()
        return {"status": "ready"}
    except FileNotFoundError as error:
        raise HTTPException(503, "Train the model first: python -m roadlab.train") from error


@app.get("/api/metadata")
def metadata():
    try:
        _, schema, metrics = resources()
    except FileNotFoundError as error:
        raise HTTPException(503, "Model artifacts missing") from error
    return {"schema": schema, "metrics": metrics, "examples": json.loads((ROOT / "artifacts/examples.json").read_text())}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        model, schema, _ = resources()
    except FileNotFoundError as error:
        raise HTTPException(503, "Model artifacts missing") from error
    try:
        records = [validate_record(record, schema) for record in request.records]
    except (ValueError, OverflowError) as error:
        raise HTTPException(422, str(error)) from error
    frame = pd.DataFrame(records, columns=FEATURES)
    probabilities = model.predict_proba(frame)
    labels = model.classes_.tolist()
    return {"predictions": [{"label": labels[int(p.argmax())], "scores": dict(zip(labels, p.tolist()))} for p in probabilities],
            "warning": "Uncalibrated model scores. Educational retrospective classification, not emergency triage or individual risk."}
