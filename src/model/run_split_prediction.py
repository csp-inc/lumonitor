import time

import argparse
import os

from azureml.core import Environment, Experiment, Run, ScriptRunConfig, Workspace
from azureml.core.runconfig import DockerConfiguration
from osgeo import gdal
import rasterio as rio

from utils import load_azml_env


def download_model_file(
    run_id: str, remote_file: str, local_file: str, experiment_name: str = "hm-2016"
) -> None:
    """Download the model.pt file for the corresponding run_id to the local
    directory. Then it is uploaded to Azure when run. Could be other ways
    to acomplish this (copy directly from run to run perhaps)"""
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = Run(experiment, run_id)
    print(model_file)
    run.download_file(remote_file, local_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir", default="conus_split_19")
    parser.add_argument("-f", "--feature_file", default="conus_hls_median_2013.vrt")
    parser.add_argument("-e", "--experiment", default="lumonitor-export")
    parser.add_argument("-m", "--model-file", default="model.pt")
    parser.add_argument("-p", "--output-prefix")
    parser.add_argument("-r", "--run-id")
    parser.add_argument("-c", "--compute-target", default="gpu-cluster")
    parser.add_argument("-g", "--num-gpus", default=20)
    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.experiment)

    remote_model_file = (
        os.path.join("outputs", args.model_file)
        if not args.model_file.startswith("outputs")
        else args.model_file
    )
    model_file = f"data/azml/{args.run_id}_{args.model_file}"
    if not os.path.exists(model_file):
        download_model_file(args.run_id, remote_model_file, model_file, args.experiment)

    split_files = os.listdir(os.path.join("data/azml", args.source_dir))
    runs = []
    for file in split_files:
        aoi = os.path.join(args.source_dir, file)
        config = ScriptRunConfig(
            docker_runtime_config=DockerConfiguration(use_docker=True),
            source_directory="./src",
            script="model/predict.py",
            compute_target=args.compute_target,
            arguments=[
                "--aoi",
                aoi,
                "--feature-file",
                args.feature_file,
                "--model-file",
                os.path.basename(model_file),
            ],
        )

        config.run_config.environment = load_azml_env(img_tag="openmpi4")
        display_name = f"{args.output_prefix} {args.run_id} {args.model_file} {file}"

        existing_runs = [
            run for run in experiment.get_runs() if run.display_name == display_name
        ]
        if len(existing_runs) == 0:
            print("no runs")
            run = experiment.submit(config)
            run.display_name = display_name
        else:
            print("run exists")
            run = existing_runs[0]

        runs.append(run)

    output_dir = f"data/predictions/{args.output_prefix}_{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        print(run.display_name)
        while run.get_status() not in ["Completed", "Failed", "Canceled"]:
            time.sleep(60)
        local_files = []
        for file in run.get_file_names():
            if file.startswith("outputs/prediction"):
                local_file = os.path.join(output_dir, os.path.basename(file))
                if not os.path.exists(local_file):
                    print(local_file)
                    run.download_file(file, output_dir)
                local_files.append(local_file)

    template = os.path.join("data/azml", args.feature_file)
    with rio.open(template) as t:
        bounds = list(t.bounds)
        xres = t.transform[0]
        yres = t.transform[5]

    vrt_file = os.path.join(output_dir, f"{args.output_prefix}_{args.run_id}.vrt")
    gdal.BuildVRT(vrt_file, local_files, outputBounds=bounds)

    mosaic_file = os.path.join(output_dir, f"{args.output_prefix}_prediction.tif")

    gdal.Warp(
        mosaic_file,
        vrt_file,
        cutlineDSName="data/azml/conus_projected.gpkg",
        outputBounds=bounds,
        xRes=xres,
        yRes=yres,
        cropToCutline=True,
        multithread=True,
        creationOptions=[
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "TILED=YES",
        ],
    )

    # Clean up to save space
    for f in local_files:
        os.remove(f)
    os.remove(vrt_file)
    os.remove(model_file)
