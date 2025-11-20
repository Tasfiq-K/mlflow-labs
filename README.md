# mlflow-labs

## Set up the environment
```bash
sudo apt update
sudo apt upgrade
sudo apt install -y python3 python3-pip python3-venv python3-full

 ## install mlflow and sickit-learn 

pip install mlflow scikit-learn pandas matplotlib seaborn
```
Using Load Balancer you will launch the MLFlow UI on 5000 port and find the experiments on experiment section.

Get the ip using
```bash
ifconfig
```
get the ip from `inet` of `wt0`

Enter the IP and and port 5000 in the load balancer

## Access MLflow UI
Go to terminal and run the MLflow server
```bash
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts '*' --cors-allowed-origins '*'
```

## Registering a Model
In this way we will create a run of a Random Forest Classifier model and will register it on Mlflow UI.
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.ensemble import RandomForestClassifier
client = mlflow.MlflowClient()

#instantiate a model
rfc = RandomForestClassifier()

#log the model
with mlflow.start_run(run_name="logging-model") as run:
    mlflow.sklearn.log_model(sk_model=rfc, artifact_path= rfc.__class__.__name__)

```
from the Mlflow UI, click on the top right Register Model button and give model name, then click Register

## Registering Using SDK(Software Development Kit)
Other than registering a model on the MLflow UI, the task can be performed through the SDK as well.

Here a RandomForestClassifier model is registered, for that we have added parameter registered_model_name in the log_model.
```python
rfc= RandomForestClassifier(n_estimators=1)
with mlflow.start_run(run_name= "registering_model_providing_name") as run:

	mlflow.sklearn.log_model(sk_model= rfc, artifact_path=rfc.__class__.__name__, registered_model_name="registered-model-sdk")

```
The model is registered with the provided name and version one.

A model can also be registered with the help of MLflow client.

```python
model_name= "registered-model-via-client"

try:
    result= client.create_registered_model(name=model_name)
except Exceptionas e:
    print(e)

print(f"Model {result.name} created")
print(f"Model description {result.description}")
print(f"Model creation timestamp {result.creation_timestamp}")
print(f"Model tags {result.tags}")
print(f"Model Alias {result.aliases}")
```
```python
try:
    model_version= client.create_model_version(
        name= model_name,
        source= run.info.artifact_uri,
        run_id= run.info.run_id,
        description= "Model version created using MLflow client")
except Exceptionas e:
    print(e)

print(f"Model version {model_version.version} created")
print(f"Model version status: {model_version.status}")
print(f"Model version description: {model_version.description}")
print(f"Model version creation timestamp: {model_version.creation_timestamp}")
print(f"Model version source: {model_version.source}")
print(f"Model version run_id: {model_version.run_id}")
print(f"Model version status message: {model_version.status_message}")
```
output:
```bash
Model version 1 created
Model version status: READY
Model version description: Model version created using MLflow client
Model version creation timestamp: 1753040532633
Model version source: file:///root/code/mlruns/0/1a050a3e748b47c188b19a20982d617d/artifacts
Model version run_id: 1a050a3e748b47c188b19a20982d617d
Model version status message:
```
## Updating Model Metadata
In this section we will update model metadata. For that we will create a new random forest classifier model.

Here we will use the client to add model description, tags, and alias.

```python
rfc= RandomForestClassifier()

registered_model_name= "random-forest classifier"
with mlflow.start_run(run_name="registering-model")as run:
    mlflow.sklearn.log_model(sk_model=rfc, artifact_path=rfc.__class__.__name__, registered_model_name=registered_model_name)

# Initializing the MLflow client    
client = mlflow.MlflowClient()

# Adding model description
client.update_registered_model(name= registered_model_name, description="This is a random forest classifier model")

# Updating model tags
registered_model_tags= {
    "project_name":"UNDERFINED",
    "task":"classification",
    "framework":"sklearn",
}
for key, value in registered_model_tags.items():
    client.set_registered_model_tag(name=registered_model_name, key=key, value=value)

# Updating model alias    
model_aliases= ["Champion", "candidate", "development"]
for model_alias in model_aliases:
    client.set_registered_model_alias(name=registered_model_name, alias= model_alias, version="1")
```
This will create a version of random-forest-classifier model.
> We can not use same alias for multiple versions of a model. It is because different versions of a model won’t be same, might vary the parameters or even the outputs like metrics and precisions. An alias serves as a unique and meaningful label like "production", "staging", or "best_model" to quickly reference a particular version. It provides an intuitive way to manage and track models without needing to remember version numbers, helping ensure consistency and clarity in deployment and experimentation.

Now we will create a second version of this model to verify that.

```python
rfc= RandomForestClassifier()

registered_model_name= "random-forest classifier"
with mlflow.start_run(run_name="registering-model")as run:
    mlflow.sklearn.log_model(sk_model=rfc, artifact_path=rfc.__class__.__name__, registered_model_name=registered_model_name)
    
# Adding alias
client.set_registered_model_alias(name=registered_model_name, alias="Champion", version="2")
```
## Add Model Version Description
Each model might contain some kind of descriptions to it

`# version 1`
```python
client.update_model_version(name= registered_model_name, version= "1", description="This is a new des for model version 1")
```
`# version 2`
```python
client.update_model_version(name=registered_model_name, version="2", description="This is a new des for model version 2")
```
This will add a description to the model version 1.

# Add Model Version Tags
```python
client.set_model_version_tag(name=registered_model_name, version="1", key="validation_status", value="pending")

client.set_model_version_tag(name=registered_model_name, version="2", key="validation_status", value="Ready for deployment")


```
Version 1 “Pending”
Version 2 "Ready for deployment"

# Get Model Version
While multiple numbers of developers are working on the same project of a model, they might need to retrieve the model version to test it out, tune it or even use it for various tasks.

```python
model_version_1 = client.get_model_version(name= registered_model_name, version="1")

print(f"Model version: {model_version_1.version}")
print(f"Model version creation time: {model_version_1.creation_timestamp}")
print(f"Model version description: {model_version_1.description}")
print(f"Model version source: {model_version_1.source}")
print(f"Model version status: {model_version_1.status}")
print(f"Model version run_id: {model_version_1.run_id}")
print(f"Model version tags: {model_version_1.tags}")
print(f"Model version aliases: {model_version_1.aliases}")
```
# Get Model Version by Aliases
As the aliases are the way of labeling a model version, it can be necessary to retrieve the version by the alias

```python
model_version_champ = client.get_model_version_by_alias(name=registered_model_name, alias="Champion")

print(f"Model version: {model_version_champ.version}")
print(f"Model version creation time: {model_version_champ.creation_timestamp}")
print(f"Model version description: {model_version_champ.description}")
print(f"Model version source: {model_version_champ.source}")
print(f"Model version status: {model_version_champ.status}")
print(f"Model version run_id: {model_version_champ.run_id}")
print(f"Model version tags: {model_version_champ.tags}")
print(f"Model version aliases: {model_version_champ.aliases}")
```
# Deleting Information or Metadata
The model information such descriptions, tags and labels can be removed if needed. Sometime the versions and even the model is needed to be removed if found better one for production.

## Delete Registered Model Version Tags
```python
client.delete_model_version_tag(name=registered_model_name, version="1", key="validation_status")
```
Deleting model alias

```python
client.delete_registered_model_alias(name=registered_model_name, alias="Champion")
```
# Delete Model Version
```python
client.delete_model_version(name=registered_model_name, version="1")
```



