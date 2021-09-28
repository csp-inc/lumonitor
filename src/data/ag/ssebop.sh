#!/usr/bin/env bash

gdalwarp -t_srs EPSG:5070 /vsizip/vsicurl/https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/yearly/downloads/y2020.zip data/ssebop_2020.tif
gdalwarp -t_srs EPSG:5070 /vsizip/vsicurl/https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/yearly/downloads/y2013.zip data/ssebop_2013.tif
gdalwarp -t_srs EPSG:5070 /vsizip/vsicurl/https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/yearly/downloads/y2016.zip data/ssebop_2016.tif
