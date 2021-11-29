#!/usr/bin/env bash

SRC_GEOJSON=$1
CONTAINER=$2
ACCOUNT_NAME=$3
ACCOUNT_KEY=$4

LAYER_NAME=$(basename $SRC_GEOJSON .geojson)
LOCAL_OUTPUT_DIR=data/tiles/$LAYER_NAME

MAX_ZOOM=11

mkdir -p $LOCAL_OUTPUT_DIR

tippecanoe --force -pk -e $LOCAL_OUTPUT_DIR/$LAYER_NAME -z $MAX_ZOOM -l $LAYER_NAME $SRC_GEOJSON

az storage blob upload-batch -d $CONTAINER -s $LOCAL_OUTPUT_DIR --account-name $ACCOUNT_NAME --account-key $ACCOUNT_KEY --max-connections 10 --content-type application/vnd.mapbox-vector-tile --content-encoding gzip
