from pathlib import Path

import typer
from loguru import logger

import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


from customer_churn_pridiction.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "clean_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    df = pd.read_csv(PROCESSED_DATA_DIR / 'cleaned_data.csv')

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    test.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)

    train.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
    df_master = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    df = df_master.copy()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') 
    encoded_features = one_hot_encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_features))

    df_encoded = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)
    df_encoded.drop(columns=['Churn'], inplace=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit(df_encoded)
    jb.dump(scaler, PROCESSED_DATA_DIR / 'standard_scaler.joblib')
    jb.dump(one_hot_encoder, PROCESSED_DATA_DIR / 'one_hot_encoder.joblib')

    logger.success("Scalar saved as scaler.joblib")


if __name__ == "__main__":
    app()
