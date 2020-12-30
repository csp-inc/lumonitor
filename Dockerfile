FROM osgeo/gdal:ubuntu-small-latest

COPY ./requirements.txt ./

RUN apt-get -qqy update \
  && curl -sSL https://sdk.cloud.google.com | bash \
  && curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash \
  && apt-get install -qqy python3-pip libspatialindex-dev \
  && python3 -m pip install -r requirements.txt
