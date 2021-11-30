import os

from azureml.core import Datastore, Workspace

Datastore.register_azure_blob_container(
    workspace=Workspace.from_config(),
    datastore_name="hls",
    container_name="hls",
    account_name=os.environ['AZURE_STORAGE_ACCOUNT'],
    account_key=os.environ['AZURE_STORAGE_ACCESS_KEY']
)

