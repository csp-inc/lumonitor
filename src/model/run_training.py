import os

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import PyTorchConfiguration

import yaml

if __name__ == "__main__":

    ws = Workspace.from_config()
    experiment = Experiment(
        workspace=ws,
        name="lumonitor-conus-impervious-2016"
    )

    distr_config = PyTorchConfiguration(node_count=15)

    config = ScriptRunConfig(
        source_directory='./src',
        script='model/train.py',
        compute_target='gpu-cluster',
        distributed_job_config=distr_config,
        arguments=[
            '--params-path',
            'model/configs/conus-impervious-2016.yml'
        ]
    )

    env = Environment("lumonitor")
    env.docker.enabled = True
    env.docker.base_image = "cspincregistry.azurecr.io/lumonitor-azml:latest"
    env.python.user_managed_dependencies = True
    env.docker.base_image_registry.address = "cspincregistry.azurecr.io"
    env.docker.base_image_registry.username = os.environ['AZURE_REGISTRY_USERNAME']
    env.docker.base_image_registry.password = os.environ['AZURE_REGISTRY_PASSWORD']

    env.environment_variables = dict(
        AZURE_STORAGE_ACCOUNT=os.environ['AZURE_STORAGE_ACCOUNT'],
        AZURE_STORAGE_ACCESS_KEY=os.environ['AZURE_STORAGE_ACCESS_KEY']
    )

    config.run_config.environment = env

    run = experiment.submit(config)
