#!/usr/bin/env bash

INPUT_FILE=$1
COLOR_FILE=$2
OUTPUT_FILE=$3

gdaldem color-relief $(realpath $INPUT_FILE) $(realpath $COLOR_FILE) $OUTPUT_FILE.3band.vrt
gdal_translate $(realpath $OUTPUT_FILE.3band.vrt) -b 1 $OUTPUT_FILE.1.vrt
gdal_translate $(realpath $OUTPUT_FILE.3band.vrt) -b 2 $OUTPUT_FILE.2.vrt
gdal_translate $(realpath $OUTPUT_FILE.3band.vrt) -b 3 $OUTPUT_FILE.3.vrt
# Stealth masking here!
gdalbuildvrt -separate $OUTPUT_FILE.vrt $(realpath $OUTPUT_FILE.1.vrt) $(realpath $OUTPUT_FILE.2.vrt) $(realpath $OUTPUT_FILE.3.vrt) $(realpath data/alpha_mask.tif)
gdal_edit.py -colorinterp_1 red -colorinterp_2 green -colorinterp_3 blue -colorinterp_4 alpha $OUTPUT_FILE.vrt
gdal_translate $OUTPUT_FILE.vrt $OUTPUT_FILE -co COMPRESS=LZW -co PREDICTOR=2 -co TILED=YES -co BIGTIFF=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256
