import re

from azureml.core import Experiment, Run, Workspace


ws = Workspace.from_config()
experiment = Experiment(
    workspace=ws, 
    name="lumonitor-conus-impervious-2016"
)

output_dir = 'data/cog/2013/prediction/conus'

id_file = 'data/slice_ids_2013.txt'
with open(id_file, 'r') as src:
    run_ids = src.read().splitlines()

for run_id in run_ids:
#for run in experiment.get_runs():
    run = Run(experiment, run_id)
#    st = run.get_details()['startTimeUtc']
    if run.status == 'Completed':# and re.search('05-23|05-22', st) is not None:
        azml_files = [
            f for f in run.get_file_names()
            if f.startswith('outputs/prediction_conus')
        ]

        if len(azml_files) > 0:
            azml_file = azml_files[0]
            print(run.id, azml_file)
            run.download_file(azml_file, output_dir)




