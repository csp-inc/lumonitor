import argparse
import os
import time

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.runconfig import PyTorchConfiguration

import yaml

if __name__ == "__main__":
    model_id = 'lumonitor-conus-impervious-2016_1620952711_8aebb74b'

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

    for f in os.listdir('data/azml/slices'):
        aoi = os.path.join('data/azml/slices/', f)
        config = ScriptRunConfig(
            source_directory='./src',
            script='model/predict2.py',
            compute_target='gpu-cluster2',
            arguments=[
                '--model_id', model_id,
                '--aoi', aoi,
            ]
        )

        config.run_config.environment = env

        run = experiment.submit(config)
#    run.wait_for_completion(wait_post_processing=True)
#    time.sleep(100)
#    print('starting copy')
#    print(run.get_file_names())
#    run.upload_folder('a/', './outputs', datastore_name='hls')

#    ds = ws.get_default_datastore()
#    ds.upload('./outputs', f'{model_id}')
#    print('done with copy')


