import argparse
import os
import time

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace

import yaml

if __name__ == "__main__":

    ws = Workspace.from_config()
    experiment = Experiment(
        workspace=ws,
        name="lumonitor-conus-impervious-2016"
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

    run_ids = []
    files = os.listdir('data/azml/slices')

    model_id = 'lumonitor-conus-impervious-2016_1620952711_8aebb74b'
    # No dir on this
    feature_file = 'conus_hls_median_2013.vrt'
    with open('data/slice_ids_2013.txt', 'a+') as dst:
        for file in files:
            print(file)
            aoi = os.path.join('model/data/azml/slices/', file)
            config = ScriptRunConfig(
                source_directory='./src',
                script='model/predict.py',
                compute_target='gpu-cluster',
                arguments=[
                    '--model_id', model_id,
                    '--aoi', aoi,
                    '--feature_file', feature_file
                ]
            )

            config.run_config.environment = env

            run = experiment.submit(config)
            run_id = run.id
            dst.write('%s\n' % run_id)
