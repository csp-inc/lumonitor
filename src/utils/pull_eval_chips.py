from azureml.core import Experiment, Run, Workspace

ws = Workspace.from_config()
experiment = Experiment(
    workspace=ws, 
    name="lumonitor-conus-impervious-2016"
)

run_id = 'lumonitor-conus-impervious-2016_1620952711_8aebb74b'

output_dir = 'docs/eval_chips'

run = Run(experiment, run_id)
for f in run.get_file_names():
    if f.endswith('png'):
        print(f)
        run.download_file(f, output_dir)
