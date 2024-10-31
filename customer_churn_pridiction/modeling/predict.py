from __future__ import annotations

import pandas as pd
import joblib
from loguru import logger
from pathlib import Path

from customer_churn_pridiction.config import PROCESSED_DATA_DIR, MODELS_DIR


class Predictor:
    def __init__(self):
        """Load the model, scaler, and encoder."""
        logger.info("Loading pre-trained scaler and one-hot encoder...")
        self.standard_scaler = joblib.load(PROCESSED_DATA_DIR / 'standard_scaler.joblib')
        self.one_hot_encoder = joblib.load(PROCESSED_DATA_DIR / 'one_hot_encoder.joblib')
        self.model = joblib.load(MODELS_DIR / 'XGBoost_model.joblib')

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame."""
        logger.info("Preprocessing input data...")
        X_encoded = self.one_hot_encoder.transform(df.select_dtypes(include=['object']))
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.one_hot_encoder.get_feature_names_out(df.select_dtypes(include=['object']).columns.tolist()))
        X_preprocessed = pd.concat([df.select_dtypes(exclude=['object']), X_encoded_df], axis=1)
        return self.standard_scaler.transform(X_preprocessed)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accepts a DataFrame and returns predictions."""
        X_scaled = self.preprocess(df)
        predictions = self.model.predict(X_scaled)
        return pd.DataFrame(predictions, columns=['Prediction'])


def main(features_path: Path, predictions_path: Path):
    predictor = Predictor()

    # Load the test features
    logger.info("Loading test features from %s...", features_path)
    test_data = pd.read_csv(features_path)

    # Perform predictions
    logger.info("Performing inference for model...")
    predictions = predictor.predict(test_data)

    # Save predictions to CSV
    logger.info("Saving predictions to %s...", predictions_path)
    predictions.to_csv(predictions_path, index=False)

    logger.success("Inference complete. Predictions saved.")


if __name__ == "__main__":
    # Example usage
    import typer
    app = typer.Typer()

    @app.command()
    def run(
        features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
        predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    ):
        main(features_path, predictions_path)

    app()
