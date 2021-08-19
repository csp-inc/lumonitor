SHELL=/usr/bin/env bash
VPATH=data/:src/data/ag/

data/irrigation_volume.tif: src/data/ag/crop_impacts.py state_irrigation_volume.tif irrigated_areas.tif
data/state_irrigation_volume.tif: irrigation_rate.gpkg cropland.tif
	python3 src/utils/rasterize_geocube.py --vector_file=$< --measurement_field=acre_feet_per_acre_irrigated --template_raster=data/cropland.tif --output_file=$@
data/irrigation_rate.gpkg: src/data/ag/irrigation_rate.R
data/irrigated_areas.tif: src/data/ag/irrigated_areas.py

data/cropland.tif: src/data/ag/cropland.py

data/%.tif: %.py
	python3 $^ $@

data/%.tif: %.R
	python3 $^ $@

data/%.gpkg: %.R
	Rscript $^ $@
