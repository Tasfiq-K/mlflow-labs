import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("winequality-red.csv")
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)  # Binary classification: quality >= 7 is "good"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("Wine_Quality_Autologging")

# Run 1: Autologging with Random Forest
mlflow.sklearn.autolog()  # Enable autologging
with mlflow.start_run(run_name="RandomForest_Autolog"):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest (Autolog) Accuracy: {rf_accuracy}")
mlflow.sklearn.autolog(disable=True)  # Disable autologging for next run

# Run 2: Manual logging with Random Forest
with mlflow.start_run(run_name="RandomForest_Manual"):
    rf_params = {"n_estimators": 100, "max_depth": 5}
    rf_model = RandomForestClassifier(**rf_params, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Manual logging
    mlflow.log_params(rf_params)
    mlflow.log_metric("accuracy", rf_accuracy)

    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Random Forest Manual Confusion Matrix")
    plt.savefig("rf_manual_confusion_matrix.png")
    mlflow.log_artifact("rf_manual_confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_manual")
    print(f"Random Forest (Manual) Accuracy: {rf_accuracy}")
