#!/usr/bin/env bash

MODEL_FILE=$1
OUTPUT_DIRECTORY=$2
CLIPPER=$3
OUTPUT_FILE=$4
INPUT_FILES="${@:5}"

mkdir -p "$OUTPUT_DIRECTORY"
# python3 src/model/predict.py -m $MODEL_FILE -o "$OUTPUT_DIRECTORY" -i $INPUT_FILES

FILES_TO_MOSAIC=$(ls -d $OUTPUT_DIRECTORY/*)

MOSAIC=temp.tif
rm -f $MOSAIC

gdal_merge.py -n nan -a_nodata nan -o $MOSAIC $FILES_TO_MOSAIC

gdalwarp -cutline $CLIPPER -crop_to_cutline $MOSAIC $OUTPUT_FILE

