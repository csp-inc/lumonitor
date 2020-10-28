## Lumonitor

Lumonitor is short for "land use monitor", but we are really only interested in one thing: human impacts.

### What's here

__src/__: Code files to collect and process data sources

### What's needed

Basically you will need:
- python3 (packages in requirements.txt)
- OGR/GDAL 3.2+ with sensible drivers

And optionally:
- GNU Make

If you are using docker (see below), then you just need docker.

#### Make

_Makefile_ contains the rules necessary to build all data to tiles using the patterns described above. Use of it is optional but encouraged. I am trying to keep any program logic out of the make recipes (and use e.g. shell scripts for multi-line recipes) so it can remain optional and hopefully clear and concise.

#### Docker

_Dockerfile_ contains the recipe for a docker image which will contain all necessary programs and libraries. __src/docker/__ contains convenience scripts which make using Docker easier. __src/docker/docker_shell.sh__ is a convenience script which runs the command in docker using the appropriate image. __src/docker/build_docker_image.sh__ builds the docker image.

#### Make and Docker

If you wish to use both `make` _and_ `docker`, simply run the corresponding make command with `-f Makefile.docker`. This is essentially a "pass-thru" makefile which calls __src/docker_shell.sh__. It also builds the docker image automatically if it has not been previously built or if the Dockerfile has been updated, tracking this through the placeholder __Dockerfile.built__.
