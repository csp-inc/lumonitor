#! /usr/bin/env bash

INPUT=/vsizip/vsicurl/https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_state_5m.zip
OUTPUT=$1
rm -f $OUTPUT

ogr2ogr -dialect sqlite -t_srs EPSG:4326 -sql "SELECT ST_Union(geometry) FROM ( SELECT geometry, LSAD FROM cb_2019_us_state_5m WHERE STUSPS NOT IN ('AK', 'GU', 'PR', 'VI', 'MP', 'AS', 'HI')) AS J GROUP BY LSAD" "$OUTPUT" "$INPUT"
