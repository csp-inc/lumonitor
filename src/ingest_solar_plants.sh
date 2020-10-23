#!/usr/bin/env bash

ogr2ogr -dialect sqlite -sql "SELECT * FROM PowerPlants_US_202004 WHERE PrimSource = 'solar'" data/solar_points.shp /vsizip/vsicurl/https://www.eia.gov/maps/map_data/PowerPlants_US_EIA.zip

gsutil cp data/solar_points.* gs://aft-saf

earthengine upload table --wait --asset_id users/jesse/lumonitor/solar_points gs://aft-saf/solar_points.shp
