#!/usr/bin/env

IMAGE_ONE=$1
IMAGE_TWO=$2
OUTPUT_IMAGE=$3

# Predictor may need to be changed depending on image
gdal_calc.py -A $IMAGE_ONE -B $IMAGE_TWO --outfile=$OUTPUT_IMAGE --calc="numpy.min((A,B), axis=0)" --co=TILED=YES --co=BLOCKXSIZE=256 --co=BLOCKYSIZE=256 --co=COMPRESS=LZW --co=PREDICTOR=2 --co=BIGTIFF=YES
