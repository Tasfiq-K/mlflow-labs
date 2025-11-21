import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.ensemble import RandomForestClassifier
client = mlflow.MlflowClient()

#instantiate a model
rfc = RandomForestClassifier()

#log the model
# with mlflow.start_run(run_name="logging-model") as run:
#     mlflow.sklearn.log_model(sk_model=rfc, artifact_path= rfc.__class__.__name__)


# register model through the SDK
# rfc= RandomForestClassifier(n_estimators=1)
# with mlflow.start_run(run_name= "registering_model_providing_name") as run:

# 	mlflow.sklearn.log_model(sk_model= rfc, name=rfc.__class__.__name__, registered_model_name="registered-model-sdk")

# through Mlflow client

# model_name= "registered-model-via-client"

# try:
#     result= client.create_registered_model(name=model_name)
# except Exception as e:
#     print(e)

# print(f"Model {result.name} created")
# print(f"Model description {result.description}")
# print(f"Model creation timestamp {result.creation_timestamp}")
# print(f"Model tags {result.tags}")
# print(f"Model Alias {result.aliases}")

# creating model version
# model_name= "registered-model-via-client"
# with mlflow.start_run(run_name= "registering_model_providing_name") as run:

# 	# mlflow.sklearn.log_model(sk_model= rfc, name=rfc.__class__.__name__, registered_model_name="registered-model-sdk")
    
    
#     try:
#         model_version = client.create_model_version(
#             name = model_name,
#             source = run.info.artifact_uri,
#             run_id = run.info.run_id,
#             description= "Model version created using MLflow client")
#     except Exception as e:
#         print(e)

#     print(f"Model version {model_version.version} created")
#     print(f"Model version status: {model_version.status}")
#     print(f"Model version description: {model_version.description}")
#     print(f"Model version creation timestamp: {model_version.creation_timestamp}")
#     print(f"Model version source: {model_version.source}")
#     print(f"Model version run_id: {model_version.run_id}")
#     print(f"Model version status message: {model_version.status_message}")

# update model mdetadata
# rfc= RandomForestClassifier()

# registered_model_name= "random-forest classifier"
# with mlflow.start_run(run_name="registering-model")as run:
#     mlflow.sklearn.log_model(sk_model=rfc, name=rfc.__class__.__name__, registered_model_name=registered_model_name)

# # Initializing the MLflow client    
# client = mlflow.MlflowClient()

# # Adding model description
# client.update_registered_model(name= registered_model_name, description="This is a random forest classifier model")

# # Updating model tags
# registered_model_tags= {
#     "project_name":"UNDERFINED",
#     "task":"classification",
#     "framework":"sklearn",
# }
# for key, value in registered_model_tags.items():
#     client.set_registered_model_tag(name=registered_model_name, key=key, value=value)

# # Updating model alias    
# model_aliases= ["Champion", "candidate", "development"]
# for model_alias in model_aliases:
#     client.set_registered_model_alias(name=registered_model_name, alias= model_alias, version="1")

# Now we will create a second version of this model to verify that.
# rfc= RandomForestClassifier()

# registered_model_name= "random-forest classifier"
# with mlflow.start_run(run_name="registering-model")as run:
#     mlflow.sklearn.log_model(sk_model=rfc, name=rfc.__class__.__name__, registered_model_name=registered_model_name)
    
# # Adding alias
# client.set_registered_model_alias(name=registered_model_name, alias="Champion", version="2")

# adding version descriptions
# version 1
# registered_model_name= "random-forest classifier"
# client.update_model_version(name= registered_model_name, version= "1", description="This is a new des for model version 1")

# # version 2
# client.update_model_version(name=registered_model_name, version="2", description="This is a new des for model version 2")

# # add model version tags
# client.set_model_version_tag(name=registered_model_name, version="1", key="validation_status", value="pending")

# client.set_model_version_tag(name=registered_model_name, version="2", key="validation_status", value="Ready for deployment")

# Get Model Version
# registered_model_name= "random-forest classifier"
# model_version_1 = client.get_model_version(name= registered_model_name, version="1")

# print(f"Model version: {model_version_1.version}")
# print(f"Model version creation time: {model_version_1.creation_timestamp}")
# print(f"Model version description: {model_version_1.description}")
# print(f"Model version source: {model_version_1.source}")
# print(f"Model version status: {model_version_1.status}")
# print(f"Model version run_id: {model_version_1.run_id}")
# print(f"Model version tags: {model_version_1.tags}")
# print(f"Model version aliases: {model_version_1.aliases}")

# get model version by alias
# registered_model_name= "random-forest classifier"
# model_version_champ =  client.get_model_version_by_alias(name=registered_model_name, alias="Champion")

# print(f"Model version: {model_version_champ.version}")
# print(f"Model version creation time: {model_version_champ.creation_timestamp}")
# print(f"Model version description: {model_version_champ.description}")
# print(f"Model version source: {model_version_champ.source}")
# print(f"Model version status: {model_version_champ.status}")
# print(f"Model version run_id: {model_version_champ.run_id}")
# print(f"Model version tags: {model_version_champ.tags}")
# print(f"Model version aliases: {model_version_champ.aliases}")

# Deleting Information or Metadat
# Delete Registered Model Version Tags
# registered_model_name= "random-forest classifier"
# client.delete_model_version_tag(name=registered_model_name, version="1", key="validation_status")

# delete model alias
# registered_model_name= "random-forest classifier"
# client.delete_registered_model_alias(name=registered_model_name, alias="Champion")

registered_model_name= "random-forest classifier"
client.delete_model_version(name=registered_model_name, version=1)


