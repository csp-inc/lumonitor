#!/usr/bin/env bash

INPUT_FILE=$1
COLOR_FILE=$2
OUTPUT_FILE=$3

gdaldem color-relief -alpha $INPUT_FILE $COLOR_FILE $OUTPUT_FILE
