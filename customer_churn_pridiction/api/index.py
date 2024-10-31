from fastapi import FastAPI, HTTPException, Body
import mlflow.pyfunc
import pandas as pd
from customer_churn_pridiction.config import MLFLOW_DIR, PROCESSED_DATA_DIR
import joblib
import numpy as np

app = FastAPI()

# Set MLflow tracking URI and load the model
mlflow.set_tracking_uri(MLFLOW_DIR)
model = mlflow.pyfunc.load_model("runs:/78389a7556d24a2d99088e3e23b323ba/model")

# Load the scaler and encoder
standard_scaler = joblib.load(PROCESSED_DATA_DIR / 'standard_scaler.joblib')
one_hot_encoder = joblib.load(PROCESSED_DATA_DIR / 'one_hot_encoder.joblib')


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """Preprocess input DataFrame for prediction."""
    X_encoded = one_hot_encoder.transform(df.select_dtypes(include=['object']))
    X_encoded_df = pd.DataFrame(X_encoded, columns=one_hot_encoder.get_feature_names_out(df.select_dtypes(include=['object']).columns.tolist()))
    X_preprocessed = pd.concat([df.select_dtypes(exclude=['object']), X_encoded_df], axis=1)
    return standard_scaler.transform(X_preprocessed)


@app.post("/predict")
def predict(input_data: list[dict] = Body(...)):  # Use Body(...) to require input_data
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(input_data)

        # Preprocess data
        X_scaled = preprocess_data(data)

        # Make prediction
        prediction = model.predict(X_scaled)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
