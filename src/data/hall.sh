#!/usr/bin/env bash

gdal_calc.py --calc='(100*(1-((1-A/100.0)*(1-B/100.0)*(1-C/100.0)))).astype(numpy.int8)' --outfile=data/predictions/hall.tiff -A data/predictions/hag_hm-2016_1636419372_96a1d581/hag_prediction_conus.tif -B data/predictions/htrans_hm-2016_1636478645_6b6a8ce5/htrans_conus_prediction.tif -C data/predictions/hurban_hm-2016_1636496718_daf497dc/hurban_conus_prediction.tif --co COMPRESS=LZW --co PREDICTOR=2 --type Byte
