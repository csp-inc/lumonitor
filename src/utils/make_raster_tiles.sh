#!/usr/bin/env bash

SRC_TIF=$1
LAYER_NAME=$(echo $(basename $SRC_TIF) | sed 's/\(.*[0-9]\).*$/\1/')

CONTAINER=tiles
# I think this will work, but untested -10/24
LOCAL_OUTPUT_DIR=data/tiles/$LAYER_NAME
TARGET=$LOCAL_OUTPUT_DIR/$LAYER_NAME
echo $TARGET

MIN_ZOOM=0
MAX_ZOOM=10

gdal2tiles.py -z $MIN_ZOOM-$MAX_ZOOM $SRC_TIF $TARGET
# copy the whole directory so subdirs are maintained.
# Has possible side effects but couldn't think of another way
az storage blob upload-batch -d $CONTAINER -s $LOCAL_OUTPUT_DIR --account-name=lumonitor --account-key=$AZURE_LUMONITOR_STORAGE_KEY --pattern *.png
