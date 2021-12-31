#!/usr/bin/env bash
AG=$1
TRANS=$2
URBAN=$3
OUTPUT=$4

OPTS="-co COMPRESS=LZW -co PREDICTOR=2 -ot Byte -co BIGTIFF=YES -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co INTERLEAVE=PIXEL"
OOPTS="--co COMPRESS=LZW --co PREDICTOR=2 --type Byte --co BIGTIFF=YES --co TILED=YES --co BLOCKXSIZE=256 --co BLOCKYSIZE=256 --co INTERLEAVE=PIXEL"


gdal_calc.py --calc='(100*(1-((1-A/100.0)*(1-B/100.0)*(1-C/100.0)))).astype(numpy.int8)' --outfile=all.tif -A $AG -B $TRANS -C $URBAN $OOPTS --overwrite --NoDataValue=127
gdalbuildvrt -separate OUTPUT.vrt all.tif $AG $TRANS $URBAN
gdalwarp -cutline data/azml/conus_projected.gpkg -crop_to_cutline $OPTS OUTPUT.vrt $OUTPUT
