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

    #    run_id = "hm-2016_1636419372_96a1d581"
    model_file = f"data/azml/{args.run_id}.pt"
    if not os.path.exists(model_file):
        download_model_file(args.run_id, model_file)

    #    distr_config = MpiConfiguration(node_count=20)
    #    config = ScriptRunConfig(
    #        source_directory="./src",
    #        script="model/predict_hvd.py",
    #        compute_target="gpu-cluster",
    #        distributed_job_config=distr_config,
    #        arguments=[
    #            "--run_id",
    #            args.run_id,
    #            "--aoi",
    #            "conus.geojson",
    #            "--feature_file",
    #            "conus_hls_median_2013.vrt",
    #        ],
    #    )
    #
    #    config.run_config.environment = load_azml_env()
    # run = experiment.submit(config)
    # run.wait_for_completion()
    run = [
        run
        for run in experiment.get_runs()
        if run.get_details()["runId"] == "hm-2016_1637291202_ae89d6b5"
    ][0]

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
    gdal.Translate(
        mosaic_file, vrt_file, creationOptions=["COMPRESS=LZW", "PREDICTOR=2"]
    )

    # Clean up to save space
    os.remove(model_file)
    for f in local_files:
        os.remove(f)
    os.remove(vrt_file)
