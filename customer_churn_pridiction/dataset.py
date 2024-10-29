from pathlib import Path

import typer
from loguru import logger
import pandas as pd

from customer_churn_pridiction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


from scipy.stats import zscore

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    df_master = pd.read_excel(RAW_DATA_DIR / "E_Commerce_Dataset.xlsx", sheet_name="E Comm")
    df = df_master.copy()
    df.drop(columns=["CustomerID"], inplace=True)
    logger.info(df.describe())
    columns_to_fill = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder"]
    numeric_cols = ['Churn', 'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderCount', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'DaySinceLastOrder', 'CashbackAmount']

    for column in columns_to_fill:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    logger.info(df.isnull().sum())
    z_scores = df[numeric_cols].apply(zscore)
    outliers_mask = (abs(z_scores) > 3.0)

    df_cleaned = df[~outliers_mask.any(axis=1)]
    df_cleaned.describe()

    logger.info(df_cleaned.corr()['Churn'].sort_values(ascending=False))
    logger.info("Data Correlation Metrix", df_cleaned.corr())
    df_cleaned.to_csv(PROCESSED_DATA_DIR / "cleaned_data.csv", index=False)

    logger.success("Processing dataset complete and saved filed.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

# selected_features = ['Tenure','Complain', 'DaySinceLastOrder', 'CashbackAmount', 'SatisfactionScore']
