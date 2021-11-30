#!/bin/bash
prefix=$1 # e.g. cog/2013/training_albers
account_name=$2
account_key=$3
output=$4
additional_files=$5

gdalbuildvrt -te -2493045.00 177285.000 2342655.000 3310005.000 -overwrite $4 $(az storage blob list --prefix=$prefix -c 'hls' --account-name=$account_name --account-key=$account_key --query []["name"] -o tsv | sed 's/^/\/vsiaz\/hls\//') $5
