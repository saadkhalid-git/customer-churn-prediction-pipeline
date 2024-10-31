from pathlib import Path

import typer

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from customer_churn_pridiction.config import PROCESSED_DATA_DIR, MLFLOW_DIR, MODELS_DIR, SHAP_PLOTS_DIR
matplotlib.use('Agg')

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    train_data = pd.read_csv(PROCESSED_DATA_DIR / 'train.csv')
    test_data = pd.read_csv(PROCESSED_DATA_DIR / 'test.csv')
    target = 'Churn'

    X_train = train_data.drop(columns=[target])
    Y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    Y_test = test_data[target]
    standard_scaler = joblib.load(MODELS_DIR / 'standard_scaler.joblib')
    one_hot_encoder = joblib.load(MODELS_DIR / 'one_hot_encoder.joblib')

    X_train_encoded = one_hot_encoder.transform(X_train.select_dtypes(include=['object']))

    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=one_hot_encoder.get_feature_names_out(X_train.select_dtypes(include=['object']).columns.tolist()))

    X_train = pd.concat([X_train.select_dtypes(exclude=['object']), X_train_encoded_df], axis=1)
    X_test_encoded = one_hot_encoder.transform(X_test.select_dtypes(include=['object']))

    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=one_hot_encoder.get_feature_names_out(X_test.select_dtypes(include=['object']).columns.tolist()))

    X_test = pd.concat([X_test.select_dtypes(exclude=['object']), X_test_encoded_df], axis=1)
    mlflow.set_tracking_uri(MLFLOW_DIR)
    mlflow.set_experiment('customer_churn_prediction_experiment')

    models = [
        ('XGBoost', xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=0)),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)),
        ('LogisticRegression', LogisticRegression(max_iter=100, random_state=0))
    ]

    # Loop through models
    for model_name, model in models:
        with mlflow.start_run(run_name=model_name):
            # Fit the model and log parameters
            model.fit(standard_scaler.transform(X_train), Y_train)
            mlflow.log_param('model', model_name)

            # Log parameters specific to the model
            model_params = {
                'XGBoost': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
                'RandomForest': {'n_estimators': 100, 'max_depth': 5},
                'LogisticRegression': {'max_iter': 100}
            }
            for param, value in model_params[model_name].items():
                mlflow.log_param(param, value)

            # Predictions and accuracy
            Y_train_pred = model.predict(standard_scaler.transform(X_train))
            Y_test_pred = model.predict(standard_scaler.transform(X_test))
            train_accuracy = accuracy_score(Y_train, Y_train_pred)
            test_accuracy = accuracy_score(Y_test, Y_test_pred)

            mlflow.log_metric('train_accuracy', train_accuracy)
            mlflow.log_metric('test_accuracy', test_accuracy)

            # Log model and artifacts
            mlflow.sklearn.log_model(model, 'model')
            mlflow.log_artifact(MODELS_DIR / 'standard_scaler.joblib')
            mlflow.log_artifact(MODELS_DIR / 'one_hot_encoder.joblib')

            # Print results
            print(f'\nModel: {model_name}')
            print(f'Train accuracy: {train_accuracy:.4f}')
            print(f'Test accuracy: {test_accuracy:.4f}')
            print('Classification report for test data:')
            print(classification_report(Y_test, Y_test_pred))

            # SHAP explanations for tree-based models only
            if model_name in ['XGBoost', 'RandomForest']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(standard_scaler.transform(X_test))

                try:
                    # Summary plot (bar)
                    plt.figure()
                    shap.summary_plot(shap_values, X_test, plot_type='bar')
                    summary_plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_summary_plot_{model_name}.png')
                    plt.savefig(summary_plot_path)
                    plt.close()  # Close the figure after saving

                    # Dependence plot
                    plt.figure()
                    shap.dependence_plot('Tenure', shap_values, X_test)
                    dependence_plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_dependence_plot_tenure_{model_name}.png')
                    plt.savefig(dependence_plot_path)
                    plt.close()  # Close the figure after saving

                    # Waterfall plot for a single instance
                    plt.figure()
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0]))
                    waterfall_plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_waterfall_plot_{model_name}.png')
                    plt.savefig(waterfall_plot_path)
                    plt.close()  # Close the figure after saving

                    # Beeswarm plot
                    plt.figure()
                    shap.summary_plot(shap_values, X_test, plot_type='dot')
                    beeswarm_plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_beeswarm_plot_{model_name}.png')
                    plt.savefig(beeswarm_plot_path)
                    plt.close()  # Close the figure after saving

                    # Force plot (specific data point)
                    force_plot_path = os.path.join(SHAP_PLOTS_DIR, f'shap_force_plot_{model_name}.html')
                    shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False)
                    plt.close()  # Close any opened figures (if applicable)

                    mlflow.log_artifact(force_plot_path)

                except Exception as e:
                    print(f"Error logging SHAP plots for {model_name}: {e}")

            # Dump model after training
            model_dump_path = f"{model_name}_model.joblib"
            joblib.dump(model, MODELS_DIR / model_dump_path)
            mlflow.log_artifact(model_dump_path)

        print(f'Model: {model_name} - completed and logged all SHAP plots and model artifacts.')


if __name__ == "__main__":
    app()
