FROM osgeo/gdal:ubuntu-small-latest

COPY ./requirements.txt ./

RUN apt-get -qqy update \
  && apt-get install -qqy  python3-pip \
  && python3 -m pip install -r requirements.txt
