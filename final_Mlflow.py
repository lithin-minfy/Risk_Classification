import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# Load dataset
df = pd.read_excel(r"C:\Users\Minfy\Downloads\Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data")

# Utility Functions
def split_data(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def perform_grid_search(pipe, X_train, y_train, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1):
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model_return_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy_score(y_test, y_pred)

# Model Pipelines
def create_rf_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

def create_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='liblinear'))
    ])

# Parameters
PARAM_GRID_RF = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

PARAM_GRID_LR = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l1', 'l2']
}

if __name__ == "__main__":
    mlflow.set_experiment("Bankloan_RF_LR_Experiment")
    mlflow.sklearn.autolog()

    TARGET_COL = 'Personal Loan'
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

    # Random Forest
    with mlflow.start_run(run_name="RandomForest") as rf_run:
        rf_pipeline = create_rf_pipeline()
        rf_grid = perform_grid_search(rf_pipeline, X_train, y_train, PARAM_GRID_RF)
        rf_accuracy = evaluate_model_return_accuracy(rf_grid.best_estimator_, X_test, y_test)

        mlflow.sklearn.log_model(
            sk_model=rf_grid.best_estimator_,
            artifact_path="rf_model"
        )
        rf_run_id = rf_run.info.run_id

    # Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression") as lr_run:
        lr_pipeline = create_logistic_pipeline()
        lr_grid = perform_grid_search(lr_pipeline, X_train, y_train, PARAM_GRID_LR)
        lr_accuracy = evaluate_model_return_accuracy(lr_grid.best_estimator_, X_test, y_test)

        mlflow.sklearn.log_model(
            sk_model=lr_grid.best_estimator_,
            artifact_path="lr_model"
        )
        lr_run_id = lr_run.info.run_id

    # Compare Accuracy
    if rf_accuracy > lr_accuracy:
        best_run_id = rf_run_id
        best_artifact = "rf_model"
        print(f"\n✅ Random Forest selected with Accuracy: {rf_accuracy:.4f}")
    else:
        best_run_id = lr_run_id
        best_artifact = "lr_model"
        print(f"\n✅ Logistic Regression selected with Accuracy: {lr_accuracy:.4f}")

    # Register best model
    model_uri = f"runs:/{best_run_id}/{best_artifact}"
    model_name = "BankloanBestModel_AutoSelected"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered as: {model_name}")

    # Load and use the best model for prediction
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")
    predictions = loaded_model.predict(X_test)
    print("\nPredictions from best model:")
    print(predictions)
