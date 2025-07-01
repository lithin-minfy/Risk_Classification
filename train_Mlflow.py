import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel(r"C:\Users\Minfy\Downloads\Bank_Personal_Loan_Modelling.xlsx", sheet_name= "Data")
df.head()

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def split_data(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_pipeline():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    return pipe

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

def evaluate_model(model, X_test, y_test):
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



import mlflow
import mlflow.sklearn

mlflow.set_experiment("Bankloan_RF_LR_Experiment")


# Set experiment name (creates it if it doesn't exist)
#mlflow.set_experiment("Loan_Approval_RF_Experiment")

# Enable autologging
mlflow.sklearn.autolog()

if __name__ == "__main__":
    with mlflow.start_run(run_name="Rf"):

        # Parameters
        TARGET_COL = 'Personal Loan'
        PARAM_GRID = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5],
            'rf__min_samples_leaf': [1, 2]
        }

        # Split the data
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

        # Create pipeline
        pipeline = create_pipeline()

        # Grid search
        grid_search = perform_grid_search(pipeline, X_train, y_train, PARAM_GRID)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")

        # Evaluate on test set
        evaluate_model(grid_search.best_estimator_, X_test, y_test)

        # Register the best model
        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_,
            artifact_path="best_model",
            registered_model_name="BankloanRandomForest"
        )



from sklearn.linear_model import LogisticRegression

# Set experiment name (creates it if it doesn't exist)
# mlflow.set_experiment("Loan_Approval_Logistic_Experiment")

# Enable autologging
mlflow.sklearn.autolog()

def create_logistic_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='liblinear'))
    ])

LOGISTIC_PARAM_GRID = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l1', 'l2']
}

if __name__ == "__main__":
    with mlflow.start_run(run_name="Lr"):

        TARGET_COL = 'Personal Loan'

        # Split the data
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)

        # Create pipeline
        logistic_pipeline = create_logistic_pipeline()

        # Grid search
        grid_search_logistic = perform_grid_search(
            logistic_pipeline, X_train, y_train, LOGISTIC_PARAM_GRID
        )

        print(f"\n[Logistic Regression] Best Parameters: {grid_search_logistic.best_params_}")
        print(f"[Logistic Regression] Best CV AUC: {grid_search_logistic.best_score_:.4f}")

        # Evaluate
        evaluate_model(grid_search_logistic.best_estimator_, X_test, y_test)

        # Register the best model
        mlflow.sklearn.log_model(
            sk_model=grid_search_logistic.best_estimator_,
            artifact_path="best_logistic_model",
            registered_model_name="BankloanLogisticModel"
        )


from mlflow.tracking import MlflowClient

# Set the experiment name
experiment_name = "Bankloan_RF_Experiment"
mlflow.set_experiment(experiment_name)

# Initialize client and get experiment ID
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Search for the best run (e.g., highest ROC AUC)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.roc_auc DESC"],  # You can change this to another metric if needed
    max_results=1
)

# Get the best run
best_run = runs[0]
best_run_id = best_run.info.run_id
print(f"Best run ID: {best_run_id}")
# print(f"ROC AUC: {best_run.data.metrics['best_cv_score']}")

# Register the model from the best run
model_uri = f"runs:/{best_run_id}/best_model"  # or "best_logistic_model" depending on your artifact path
model_name = "BankloanBestModel_again"

mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as: {model_name}")


import mlflow.pyfunc
import pandas as pd

model_name = "BankloanBestModel"
model_stage = "None"  # or "Staging", "Production", etc.

model_uri = f"models:/{model_name}/latest"  # or specify version like 'models:/LoanApprovalBestModel/1'
model = mlflow.pyfunc.load_model(model_uri)



X_train, X_test, y_train, y_test = split_data(df, TARGET_COL)
predictions = model.predict(X_test)

# View results
print(predictions)