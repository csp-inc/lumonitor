#!/usr/bin/env bash

gdal_rasterize -burn 255 -init 0 -te -2493045.00 177285.000 2342655.000 3310005.000 -ts 154535 97541 -ot Byte -co COMPRESS=LZW -co PREDICTOR=2 data/azml/conus_projected.gpkg data/alpha_mask.tif
