#!/usr/bin/env bash

SRC_TIF=$1
LAYER_NAME=$2

CONTAINER=tiles
LOCAL_OUTPUT_DIR=data/tiles/$LAYER_NAME/$LAYER_NAME


SRC_STEM=$(basename $SRC_TIF .tif)
LAYER_NAME=$(basename $TARGET_PLACEHOLDER .tiles)
TARGET=$LOCAL_OUTPUT_DIR$LAYER_NAME
echo $TARGET

MIN_ZOOM=0
MAX_ZOOM=10

gdal2tiles.py --processes=8 -r cubic -z $MIN_ZOOM-$MAX_ZOOM $SRC_TIF $TARGET

# copy the whole directory so subdirs are maintained.
# Has possible side effects but couldn't think of another way
# Well here's 1: make a nested self-named directory.
# Ok maybe tomorrow.
# az storage blob upload-batch -d $CONTAINER -s $LOCAL_OUTPUT_DIR
