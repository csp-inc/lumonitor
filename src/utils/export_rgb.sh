#!/usr/bin/env bash

INPUT_FILE=$1
COLOR_FILE=$2
OUTPUT_FILE=$3
TMPFILE=$(mktemp -u).tif

# Havta do it here, but should be done in prediction
gdalwarp -cutline data/azml/conus_projected.gpkg -crop_to_cutline $INPUT_FILE $TMPFILE
gdaldem color-relief -co COMPRESS=LZW -co PREDICTOR=2 -alpha $TMPFILE $COLOR_FILE $OUTPUT_FILE
rm $TMPFILE
