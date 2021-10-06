SHELL=/usr/bin/env bash
VPATH=data/:src/data/ag/

data/net_water_2020.tif: daymet_precip_2020.tif ssebop_2020.tif

data/daymet_precip_2020.tif: irrigated_areas.tif
	python src/data/ag/daymet.py

data/ssebop_2020.tif:
	source src/data/ag/ssebop.sh

data/irrigation_volume.tif: irrigated_areas.tif irrigation_rate.gpkg
	gdal_translate -ot Float32 -co COMPRESS=LZW -co PREDICTOR=3 $< $@
	gdal_rasterize -a acre_feet_per_pixel data/irrigation_rate.gpkg temp.tif
	# Likely another way to do this I haven't discovered, just need to essentially
	# mask temp.tif to irrigated areas
	gdal_calc.py --calc="A*B" --outfile=$@ -A $< -B temp.tif --co=COMPRESS=LZW --co=PREDICTOR=3

data/irrigation_rate.gpkg: src/data/ag/irrigation_rate.R
data/irrigated_areas.tif: src/data/ag/irrigated_areas.py

data/cropland.tif: src/data/ag/cropland.py

data/%.tif: %.py
	python3 $^ $@

data/%.tif: %.R
	python3 $^ $@

data/%.gpkg: %.R
	Rscript $^ $@
