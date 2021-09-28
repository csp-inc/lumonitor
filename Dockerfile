FROM osgeo/gdal:ubuntu-small-latest

ARG requirements_file

COPY ./$requirements_file ./requirements.txt

RUN apt-get -qqy update \

  && sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' \
  && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
  && apt-get -qqy update \
# Install the latest version of PostgreSQL.
# If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
  && apt-get -y install postgresql \
  && curl -sSL https://sdk.cloud.google.com | bash \
  && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
  && apt-get install -qqy python3-pip libspatialindex-dev git \
  && python3 -m pip install -r requirements.txt
