#!/bin/bash 
year=$1
account_name=$2
account_key=$3
output=$4

gdalbuildvrt -te -2493045.00 177285.000 2342655.000 3310005.000 -overwrite $4 $(az storage blob list --prefix=cog/$year/training_albers -c 'hls' --account-name=$account_name --account-key=$account_key --query []["name"] -o tsv | sed 's/^/\/vsiaz\/hls\//')
