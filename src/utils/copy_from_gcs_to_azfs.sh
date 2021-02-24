#!/usr/bin/env

INPUT_GLOB=$1
OUTPUT_CONTAINER=$2

LOCAL_TMP_DIR=gcs_to_az_tmp
mkdir -p $LOCAL_TMP_DIR

gsutil -m cp $INPUT_GLOB $LOCAL_TMP_DIR
az storage blob upload-batch -d $OUTPUT_CONTAINER -s $LOCAL_TMP_DIR

#rm -rf $LOCAL_TMP_DIR
