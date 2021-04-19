import os

from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="lumonitor-conus-impervious-2016")

config = ScriptRunConfig(
    source_directory='./src',
    script='model/train_azml.py',
    compute_target='gpu-cluster',
)

env = Environment("lumonitor")
env.docker.enabled = True
env.docker.base_image = "cspincregistry.azurecr.io/fd-openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04:latest"
# check this
# env.python.user_managed_dependencies = True
env.docker.base_image_registry.address = "cspincregistry.azurecr.io"
env.docker.base_image_registry.username = os.environ['AZURE_REGISTRY_USERNAME']
env.docker.base_image_registry.password = os.environ['AZURE_REGISTRY_PASSWORD']

conda_dep = CondaDependencies()
conda_dep.add_conda_package("geopandas")
env.python.conda_dependencies = conda_dep

env.environment_variables = dict(
    AZURE_STORAGE_ACCOUNT=os.environ['AZURE_STORAGE_ACCOUNT'],
    AZURE_STORAGE_ACCESS_KEY=os.environ['AZURE_STORAGE_ACCESS_KEY']
)

config.run_config.environment = env

run = experiment.submit(config)

url = run.get_portal_url()
print(url)
