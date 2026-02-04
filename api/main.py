from fastapi import FastAPI
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from clearml import Task
from api.schema import CreditInput

app = FastAPI(title="Credit Scoring API")

# -------------------------------
# Load artifacts from training task
# -------------------------------
TRAINING_TASK_ID = "dd91465cfbeb426c925678ebccd9676d"

task = Task.get_task(task_id=TRAINING_TASK_ID)

# Preprocessor (artifact)
preprocessor_path = task.artifacts["preprocessor"].get_local_copy()
preprocessor = joblib.load(preprocessor_path)

# CatBoost model (artifact)
model_path = task.artifacts["catboost_model"].get_local_copy()
model = CatBoostRegressor()
model.load_model(model_path)

# -------------------------------
# Risk banding
# -------------------------------
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

# -------------------------------
# API Endpoints
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Scoring API is running"}

@app.post("/predict")
def predict(data: CreditInput):
    df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(df)
    pd_score = float(model.predict(processed)[0])

    return {
        "probability_default": pd_score,
        "risk_band": risk_band(pd_score)
    }
