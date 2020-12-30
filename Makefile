# Load variables in .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

VPATH=src/:data/
SHELL=/usr/bin/env bash
BUCKET=gs://lumonitor/

data/training_data.azfs: training_data.gcs
	source src/copy_from_gcs_to_azfs.sh $(BUCKET)sample_data* lumonitor
	touch $@

data/training_data.gcs: get_training_data.py training_points.ee
	python $^
	touch $@

data/training_points.ee: training_points.csv
	gsutil cp $^ $(BUCKET)
	earthengine upload table -f -w --x_column lon --y_column lat --asset_id users/jesse/lumonitor/training_points $(BUCKET)$(notdir $^)
	touch $@

data/training_points.csv: make_training_points.R aoi.gpkg
	Rscript $^ $@

data/aoi.gpkg: make_aoi.sh
	source $^ $@
