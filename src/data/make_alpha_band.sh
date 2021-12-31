#!/usr/bin/env bash
# The extent here matches the prediction outputs, not the inputs
gdal_rasterize -burn 255 -init 0 -te -2366985.000 257205.000 2269065.000 3183435.000 -ts 154535 97541 -ot Byte -co COMPRESS=LZW -co PREDICTOR=2 data/azml/conus_projected.gpkg data/alpha_mask.tif
