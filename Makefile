# Load variables in .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

DATA=data/
BLOB_DIR=$(DATA)blobs
COG_DIR=$(DATA)cog

VPATH=src/:$(DATA):$(BLOB_DIR):$(COG_DIR)
SHELL=/usr/bin/env bash

# I really am not sure why this works, some interaction between --delimiter and
# -o tsv makes it list just the paths. Alternatively use jq & json output, I guess.
BLOBS=$(shell az storage blob list --delimiter 'zarr' -c lumonitor --account-key=$(AZURE_STORAGE_KEY) --account-name=$(AZURE_STORAGE_ACCOUNT) -o tsv | grep 'zarr' | tr '\n' ' ')

#GCS=$(shell gsutil ls gs://lumonitor/hls_tiles/)

#GCS_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(notdir $(GCS)))
#LABEL_COGS=$(patsubst %.blob,%, $(GCS_PLACEHOLDERS))

BLOB_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(BLOBS))
FEATURE_COGS=$(patsubst data/blobs/%.zarr.blob,data/cog/%.tif, $(BLOB_PLACEHOLDERS))
$(info $(FEATURE_COGS))

.PHONY: all cogs predictions tiles
all: cogs
cogs: $(FEATURE_COGS)

$(BLOB_PLACEHOLDERS): %:
$(FEATURE_COGS): %.tif: | %.zarr.blob

data/cog/%.tif: data/blobs/%.tif.gcs

data/cog/%.tif: data/blobs/%.zarr.blob
	python3 src/utils/zarr_to_cog.py az://lumonitor/$*.zarr $@ --account-name=$(AZURE_STORAGE_ACCOUNT) --account-key=$(AZURE_STORAGE_KEY)

data/%.gcs:
	touch $@

data/%.blob: 
	touch $@
