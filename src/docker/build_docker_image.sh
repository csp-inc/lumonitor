#!/usr/bin/env bash

DOCKERFILE="$1"
REQUIREMENTS_FILE="$2"
TAG="${3,,}"
TAG_FILE="$4"

echo "$TAG"

mkdir r-build-dir
cp $DOCKERFILE r-build-dir/Dockerfile
cp $REQUIREMENTS_FILE r-build-dir
cd r-build-dir
docker build . -t $TAG --build-arg requirements_file="$REQUIREMENTS_FILE"
cd ..
rm -rf r-build-dir

if [ "$#" -eq "3" ]
then
  echo $TAG
  echo $TAG > $TAG_FILE
fi
