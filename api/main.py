from fastapi import FastAPI
import pandas as pd
import joblib
import os

from catboost import CatBoostRegressor
from clearml import Task
from api.schema import CreditInput

# FastAPI app
app = FastAPI(title="Credit Scoring API")

# Global objects (loaded once)
model = None
preprocessor = None

# Startup: load model from ClearML
@app.on_event("startup")
def startup_event():
    global model, preprocessor

    TRAINING_TASK_ID = os.getenv("TRAINING_TASK_ID")
    if not TRAINING_TASK_ID:
        raise RuntimeError("TRAINING_TASK_ID environment variable is not set")

    task = Task.get_task(task_id=TRAINING_TASK_ID)

    # Load preprocessor
    preprocessor_path = task.artifacts["preprocessor"].get_local_copy()
    preprocessor = joblib.load(preprocessor_path)

    # Load CatBoost model
    model_path = task.artifacts["catboost_model"].get_local_copy()
    model = CatBoostRegressor()
    model.load_model(model_path)

# Risk banding
def risk_band(pd_score: float) -> str:
    if pd_score < 0.02:
        return "Very Low"
    elif pd_score < 0.05:
        return "Low"
    elif pd_score < 0.10:
        return "Medium"
    elif pd_score < 0.20:
        return "High"
    else:
        return "Very High"


# API Endpoints
@app.get("/")
def home():
    return {"message": "Credit Scoring API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(df)
    pd_score = float(model.predict(processed)[0])

    return {
        "probability_default": pd_score,
        "risk_band": risk_band(pd_score)
    }
