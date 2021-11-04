import argparse
import os
import yaml

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import MpiConfiguration, PyTorchConfiguration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of experiment")
    parser.add_argument(
        "--params-path", help="Path to params file, see src/model/configs"
    )

    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.name)

    # Since for the training script directories are relative to the training script
    params_path = os.path.join("src", args.params_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    node_count = params["num_gpus"] if params["use_hvd"] else 1
    # distr_config = PyTorchConfiguration(node_count=node_count)
    distr_config = MpiConfiguration(node_count=node_count)

    config = ScriptRunConfig(
        source_directory="./src",
        script="model/train.py",
        compute_target="gpu-cluster",
        distributed_job_config=distr_config,
        arguments=["--params-path", args.params_path],
    )

    env = Environment("lumonitor")
    env.docker.enabled = True
    env.docker.base_image = "cspincregistry.azurecr.io/lumonitor-azml:latest"
    env.python.user_managed_dependencies = True
    env.docker.base_image_registry.address = "cspincregistry.azurecr.io"
    env.docker.base_image_registry.username = os.environ["AZURE_REGISTRY_USERNAME"]
    env.docker.base_image_registry.password = os.environ["AZURE_REGISTRY_PASSWORD"]

    env.environment_variables = dict(
        AZURE_STORAGE_ACCOUNT=os.environ["AZURE_STORAGE_ACCOUNT"],
        AZURE_STORAGE_ACCESS_KEY=os.environ["AZURE_STORAGE_ACCESS_KEY"],
    )

    config.run_config.environment = env

    run = experiment.submit(config)
