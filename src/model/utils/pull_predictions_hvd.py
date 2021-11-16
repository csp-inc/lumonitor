import os

import argparse
from azureml.core import Experiment, Run, Workspace
from osgeo import gdal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-id")
    parser.add_argument("-p", "--output-prefix")
    parser.add_argument("-e", "--experiment", default="hm-2016")
    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name=args.experiment)
    run = Run(experiment, args.run_id)

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
