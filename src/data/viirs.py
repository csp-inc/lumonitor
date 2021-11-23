#!/usr/bin/env bash

INPUT_NL=$1
TEMPLATE=$2
OUTPUT=$3

gdalwarp -t_srs <(gdalsrsinfo -o wkt $TEMPLATE) -te -2493045.00 177285.000 2342655.000 3310005.000 -co COMPRESS=LZW -co PREDICTOR=3
