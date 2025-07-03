import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.pipeline.column_mapping import ColumnMapping
from sklearn.model_selection import train_test_split
import mlflow
import os

# --------------------- Metric Logging Function ---------------------
def log_evidently_metrics(report_dict, prefix=""):
    for metric in report_dict['metrics']:
        metric_name = metric.get('metric', 'unknown').replace('.', '_')
        result = metric.get('result', {})
        if isinstance(result, dict):
            if 'drift_by_columns' in result:
                for col, val in result['drift_by_columns'].items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(f"{prefix}drift_{col}", val)
            for k, v in result.items():
                if k == 'drift_by_columns':
                    continue
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}{metric_name}_{k}", v)
                elif isinstance(v, bool):
                    mlflow.set_tag(f"{prefix}{metric_name}_{k}", str(v))

# -------------------------- Main Function --------------------------
def run_evidently_drift_reports():
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(base_dir, "../uploads")
    historical_path = os.path.join(uploads_dir, "Bank_Personal_Loan.csv")
    new_data_path = os.path.join(uploads_dir, "New_data.csv")

    # Load historical data
    df = pd.read_csv(historical_path)
    X = df.drop(columns=["Personal Loan", "ID"])
    y = df["Personal Loan"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_df = X_train.copy()
    train_df["target"] = y_train
    test_df = X_test.copy()
    test_df["target"] = y_test

    # Load new data
    new_df = pd.read_csv(new_data_path)
    if "Personal Loan" in new_df.columns:
        new_df = new_df.rename(columns={"Personal Loan": "target"})
    else:
        new_df["target"] = -1  # Dummy

    # Save split data to uploads folder
    train_data_path = os.path.join(uploads_dir, "train_data.csv")
    test_data_path = os.path.join(uploads_dir, "test_data.csv")
    new_data_path_final = os.path.join(uploads_dir, "current_data.csv")

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    new_df.to_csv(new_data_path_final, index=False)

    # Column Mapping
    column_mapping = ColumnMapping(
        target="target",
        prediction=None,
        numerical_features=X.select_dtypes(include='number').columns.tolist(),
        categorical_features=X.select_dtypes(include=['object', 'category']).columns.tolist()
    )

    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("LoanDriftMonitoring")

    # ---------------- Run 1: Old vs New ----------------
    with mlflow.start_run(run_name="Drift-HistoricvsNew"):
        report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
        report.run(reference_data=train_df, current_data=new_df, column_mapping=column_mapping)

        html_path = os.path.join(base_dir, "report_old_vs_new.html")
        report.save_html(html_path)
        mlflow.log_artifact(html_path, artifact_path="evidently_reports")

        mlflow.log_artifact(train_data_path, artifact_path="data")
        mlflow.log_artifact(new_data_path_final, artifact_path="data")

        log_evidently_metrics(report.as_dict(), prefix="historic_")
        mlflow.log_metric("rows_train", len(train_df))
        mlflow.log_metric("rows_new", len(new_df))
        mlflow.log_param("drift_report", "Old vs New")

    mlflow.end_run()

    # ---------------- Run 2: Train vs Test ----------------
    mlflow.start_run(run_name="Drift-TrainvsTest")

    report_tt = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    report_tt.run(reference_data=train_df, current_data=test_df, column_mapping=column_mapping)

    html_path_tt = os.path.join(base_dir, "train_vs_test_report.html")
    report_tt.save_html(html_path_tt)
    mlflow.log_artifact(html_path_tt, artifact_path="evidently_reports")

    mlflow.log_artifact(train_data_path, artifact_path="data")
    mlflow.log_artifact(test_data_path, artifact_path="data")

    log_evidently_metrics(report_tt.as_dict(), prefix="train_test_")
    mlflow.log_metric("rows_train", len(train_df))
    mlflow.log_metric("rows_test", len(test_df))
    mlflow.log_param("drift_report", "Train vs Test")

    mlflow.end_run()

    print("✅ Both drift reports generated and logged to MLflow.")

# -------------------------- Execute Script --------------------------
if __name__ == "__main__":
    run_evidently_drift_reports()
