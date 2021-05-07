#!/usr/bin/env bash

DOCKERFILE="$1"
REQUIREMENTS_FILE="$2"
TAG="${3,,}"
TAG_FILE="$4"
BUILD_TYPE="$5"

if [ "$BUILD_TYPE" = "acr" ]
then
  REGISTRY=$(echo $TAG | cut -d'.' -f1)
  CMD="source az acr build . -r $REGISTRY"
else
  CMD="docker build ."
fi

echo "$TAG"

mkdir r-build-dir
cp $DOCKERFILE r-build-dir/Dockerfile
cp $REQUIREMENTS_FILE r-build-dir
cd r-build-dir
$CMD -t $TAG --build-arg requirements_file="$REQUIREMENTS_FILE"
cd ..
rm -rf r-build-dir

if [ "$#" -eq "4" ]
then
  echo $TAG
  echo $TAG > $TAG_FILE
fi
