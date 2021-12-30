#!/usr/bin/env bash
AG=$1
TRANS=$2
URBAN=$3
OUTPUT=$4

gdal_calc.py --calc='(100*(1-((1-A/100.0)*(1-B/100.0)*(1-C/100.0)))).astype(numpy.int8)' --outfile="$OUTPUT" -A "$AG" -B "$TRANS" -C "$URBAN" --co COMPRESS=LZW --co PREDICTOR=2 --type Byte
