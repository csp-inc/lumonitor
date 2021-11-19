#!/usr/bin/env bash
# This script blends 2 RGB images using the "lighten" algorithm, or by taking the
# maximum value of each band for each cell. The fourth band is needed to preserve
# transparency in masked areas and is the "official" no data value, hence the removal
# of the one set by gdal_calc (which is 255)

IMG_A=$1
IMG_B=$2
OUTPUT_IMG=$3

gdal_calc.py --calc="maximum(A,B)" --calc="maximum(C,D)" --calc="maximum(E,F)" --calc="maximum(G,H)" -A $IMG_A --A_band=1 -B $IMG_B --B_band=1 -C $IMG_A --C_band=2 -D $IMG_B --D_band=2 -E $IMG_A --E_band=3 -F $IMG_B --F_band=3 -G $IMG_A --G_band=4 -H $IMG_B --H_band=4 --outfile=$OUTPUT_IMG --co=COMPRESS=LZW --co=PREDICTOR=2 --co=BIGTIFF=YES --overwrite

# Gdal >= 3.3 can use –NoDataValue=none above but
gdal_edit.py $OUTPUT_IMG -unsetnodata
