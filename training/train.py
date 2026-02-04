from clearml import Task
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from utils.winsorizer import Winsorizer


def main():
    task = Task.init(
        project_name="Credit Scoring",
        task_name="PD Model Training - CatBoost Champion"
    )

    data = pd.read_csv("data/loan_applicant_data.csv")

    # Drop leakage / IDs
    data = data.drop(columns=["CustomerID", "LoanID"])

    # Structural missing values
    data["IsBusiness"] = (data["ApplicantType"] == "Business").astype(int)
    business_cols = ["BusinessRevenue", "ProfitMargin", "BusinessYears"]
    data[business_cols] = data[business_cols].fillna(0)

    X = data.drop(columns=["ProbDefault"])
    y = data["ProbDefault"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = ["ApplicantType", "EmploymentType", "LoanType"]

    num_pipeline = Pipeline([
        ("winsorize", Winsorizer()),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )

    model.fit(X_train_p, y_train)

    joblib.dump(preprocessor, "models/preprocessor.pkl")
    model.save_model("models/catboost_model.cbm")

    task.upload_artifact("preprocessor", "models/preprocessor.pkl")
    task.upload_artifact("catboost_model", "models/catboost_model.cbm")


if __name__ == "__main__":
    main()
