#!/usr/bin/env bash

if [  "$1" == "-c" ] ; then
    shift
fi

LAST_ARG="${!#}" 

# If the last argument is a file, then the image name is the content of
# that file. Otherwise, the last argument is the image name.
if [ -f "$LAST_ARG" ]; then
  IMG="$(<$LAST_ARG)"
else
  IMG="$LAST_ARG"
fi

ALL_ARGS_BUT_LAST="${*%${!#}}"

docker run -it \
  -v `pwd`:`pwd` \
  -w `pwd` \
  -it $IMG  \
  /bin/bash -c $ALL_ARGS_BUT_LAST
