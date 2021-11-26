import argparse
import os

from azureml.core import Environment, Experiment, Run, ScriptRunConfig, Workspace
from azureml.core.runconfig import MpiConfiguration
from osgeo import gdal

from utils import load_azml_env


def download_model_file(run_id: str, local_file: str) -> None:
    """Download the model.pt file for the corresponding run_id to the local
    directory. Then it is uploaded to Azure when run. Could be other ways
    to acomplish this (copy directly from run to run perhaps)"""
    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name="hm-2016")
    run = Run(experiment, run_id)
    azml_file = "outputs/model.pt"
    run.download_file(azml_file, local_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-id")
    parser.add_argument("-p", "--output-prefix")
    parser.add_argument("-f", "--feature_file", default="conus_hls_median_2013.vrt")
    parser.add_argument("-e", "--experiment", default="hm-2016")
    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.experiment)

    model_file = f"data/azml/{args.run_id}.pt"
    if not os.path.exists(model_file):
        download_model_file(args.run_id, model_file)

    distr_config = MpiConfiguration(node_count=20)
    config = ScriptRunConfig(
        source_directory="./src",
        script="model/predict.py",
        compute_target="gpu-cluster",
        distributed_job_config=distr_config,
        arguments=[
            "--run_id",
            args.run_id,
            "--aoi",
            "conus.geojson",
            "--feature_file",
            args.feature_file,
        ],
    )

    config.run_config.environment = load_azml_env()
    display_name = f"{args.output_prefix} {args.run_id}"

    existing_runs = [
        run for run in experiment.get_runs() if run.display_name == display_name
    ]
    if len(existing_runs) == 0:
        print("no runs")
        run = experiment.submit(config)
        run.display_name = display_name
        run.wait_for_completion()
    else:
        print("run exists")
        run = existing_runs[0]

    output_dir = f"data/predictions/{args.output_prefix}_{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)

    local_files = []
    for file in run.get_file_names():
        if file.startswith("outputs/prediction_conus"):
            local_file = os.path.join(output_dir, os.path.basename(file))
            if not os.path.exists(local_file):
                print(local_file)
                run.download_file(file, output_dir)
            local_files.append(local_file)

    vrt_file = os.path.join(output_dir, f"{args.output_prefix}_{args.run_id}.vrt")
    gdal.BuildVRT(vrt_file, local_files)

    mosaic_file = os.path.join(output_dir, f"{args.output_prefix}_prediction_conus.tif")
    # If you're in here, change this to warp and crop to conus
    gdal.Warp(
        mosaic_file,
        vrt_file,
        cutlineDSName="data/azml/conus_projected.gpkg",
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
