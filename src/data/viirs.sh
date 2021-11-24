#!/usr/bin/env bash

INPUT_NL=$1
TEMPLATE=$2
CROPPER=$3
OUTPUT=$4

T_SRS=$(echo $(gdalsrsinfo -o wkt $TEMPLATE)| tr -d '\n')
gdalwarp -overwrite -cutline $CROPPER -crop_to_cutline -dstnodata -1000 -t_srs "$T_SRS" -te -2493045.00 177285.000 2342655.000 3310005.000 -co COMPRESS=LZW -co PREDICTOR=3 $INPUT_NL $OUTPUT
