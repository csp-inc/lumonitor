#!/usr/bin/env bash

DOCKERFILE="$1"
TAG="${2,,}"
TAG_FILE="$3"

mkdir r-build-dir
cp $DOCKERFILE r-build-dir/Dockerfile
cp requirements.txt r-build-dir
cd r-build-dir
docker build . -t $TAG
cd ..
rm -rf r-build-dir

if [ "$#" -eq "3" ]
then
  echo $TAG
  echo $TAG > $TAG_FILE
fi
