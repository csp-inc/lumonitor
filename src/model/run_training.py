import argparse
import os
import yaml

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import MpiConfiguration

from utils import load_azml_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of experiment")
    parser.add_argument(
        "--params-path", help="Path to params file, see src/model/configs"
    )
    parser.add_argument(
        "--compute-target", help="Which cluster to use", default="gpu-cluster"
    )
    parser.add_argument("--run-label", help="label for the run", required=False)
    #    parser.add_argument("--num_gpus", help="Number of GPUs to use", default=20)

    # Just be aware that there are args passed on to train.py which will not
    # show up if you view the cl help
    args, args_to_pass_on = parser.parse_known_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.name)

    # Since for the training script directories are relative to the training script
    params_path = os.path.join("src", args.params_path)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    params.update(vars(args))

    node_count = params["num_gpus"] if params["use_hvd"] else 1

    distr_config = MpiConfiguration(node_count=node_count)

    config = ScriptRunConfig(
        source_directory="./src",
        script="model/train.py",
        compute_target=args.compute_target,
        distributed_job_config=distr_config,
        arguments=["--params-path", args.params_path] + args_to_pass_on,
    )

    config.run_config.environment = load_azml_env()

    run = experiment.submit(config)
    if args.run_label is not None:
        run.display_name = args.run_label
