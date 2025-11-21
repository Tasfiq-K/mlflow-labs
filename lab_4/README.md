# MLflow Autologging

In this hands on lab we will learn two main ways to log model training metadata, autologging and manual logging.

## Objective
Learn to use MLflow’s autolog() feature to automatically log parameters, metrics, and models,
To compare autologging with manual logging using the Wine Quality dataset.

### What is Autologging?
Autologging automatically captures information from popular ML libraries (e.g., scikit-learn, tensorflow, xgboost, pytorch, etc.) without explicitly writing logging code.

How is this different from Manual Logging? In manual logging you explicitly log each artifact, parameter, and metric using the mlflow.log_*() functions inside an mlflow.start_run() context.

## Step 1: Set Up the Environment
1. Update the system and install dependencies

Open a terminal and run:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```
2. 

```bash
mkdir wine_quality_autologging
cd wine_quality_autologging
python3 -m venv mlflow_autolog_env
source mlflow_autolog_env/bin/activate
```
3. Install Mlflow, scikit-learn and pandas
```bash
pip install mlflow scikit-learn pandas
```
4. Ensure the Wine Quality dataset is available

* If you have winequality-red.csv from the previous lab, copy it to the wine_quality_autologging directory:
    ```bash
    cp ../wine_quality_experiment/winequality-red.csv .
    ```
* Alternatively, download it using the Kaggle API:
    ```bash
    pip install kaggle
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_api_key
    kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009
    unzip red-wine-quality-cortez-et-al-2009.zip
    ```
# Set up the Environment
```bash
sudo apt update
sudo apt upgrade
sudo apt install -y python3 python3-pip python3-venv python3-full

 ## install mlflow and sickit-learn 

pip install mlflow scikit-learn pandas matplotlib seaborn
```
## Step 2: Start the MLflow Tracking Server
Using Load Balancer you will launch the MLFlow UI on 5000 port and find the experiments on experiment section.

Get the ip using
```bash
ifconfig
```
get the `inet` from the `wt0` and create the load balancer providing the ip and port 5000, then expose the port

## Access MLflow UI
Go to terminal and run the MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts '*' --cors-allowed-origins '*'
```
## Step 3: Implement Autologging and Manual Logging
1. Create a Python script for autologging and manual logging

Create a file named wine_quality_autologging.py with the following code:
```python
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
```
run the script and compare the logs side by side on the mlflow ui



