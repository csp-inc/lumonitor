#!/usr/bin/env bash

PIECE_DIRECTORY=$1
CLIPPER=$2
OUTPUT_FILE=$3

FILES_TO_MOSAIC=$(ls -d $PIECE_DIRECTORY/*)

MOSAIC=temp.tif
rm -f $MOSAIC

#gdal_merge.py -n nan -a_nodata nan -o $MOSAIC $FILES_TO_MOSAIC
gdal_merge.py -n 127 -a_nodata 127 -co COMPRESS=LZW -co PREDICTOR=2 -co BIGTIFF=YES -o $MOSAIC $FILES_TO_MOSAIC

gdalwarp -cutline $CLIPPER -crop_to_cutline -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co COMPRESS=LZW -co PREDICTOR=2 -co BIGTIFF=YES $MOSAIC $OUTPUT_FILE

