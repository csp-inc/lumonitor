# Load variables in .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

CONTAINER=hls
ACCOUNT_KEY=$(AZURE_EASTUS2_STORAGE_KEY)
ACCOUNT_NAME=$(AZURE_EASTUS2_STORAGE_ACCOUNT)

DATA=data/
BLOB_DIR=$(DATA)blobs
COG_DIR=$(DATA)cog

VPATH=src/:$(DATA):$(BLOB_DIR):$(COG_DIR)
SHELL=/usr/bin/env bash

PREFIX=2016/

# I really am not sure why this works, some interaction between --delimiter and
# -o tsv makes it list just the paths. Alternatively use jq & json output, I guess.
BLOBS=$(shell az storage blob list --prefix zarr/$(PREFIX) --delimiter 'zarr' -c $(CONTAINER) --account-key=$(ACCOUNT_KEY) --account-name=$(ACCOUNT_NAME) -o tsv | sed 's/.zarr\//.zarr/' | tr '\n' ' ')

#$(info $(BLOBS))

#GCS=$(shell gsutil ls gs://lumonitor/hls_tiles/)

#GCS_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(notdir $(GCS)))
#LABEL_COGS=$(patsubst %.blob,%, $(GCS_PLACEHOLDERS))

BLOB_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(BLOBS))
FEATURE_COG_PLACEHOLDERS=$(patsubst data/blobs/zarr/$(PREFIX)%.zarr.blob,data/blobs/cog/$(PREFIX)%.tif.blob, $(BLOB_PLACEHOLDERS))

.PHONY: all cogs predictions tiles
all: cogs
cogs: $(FEATURE_COG_PLACEHOLDERS)

data/blobs/cog/$(PREFIX)%.tif.blob: data/blobs/zarr/$(PREFIX)%.zarr.blob
	python3 src/utils/zarr_to_cog.py  --account-name=$(ACCOUNT_NAME) --account-key=$(ACCOUNT_KEY) az://$(CONTAINER)/zarr/$(PREFIX)$*.zarr /vsiaz/hls/cog/2016/training/$(*F).tif
	mkdir -p $(@D)
	touch $@

#data/%.gcs:
#	echo touch $@

data/%.zarr.blob: 
	mkdir -p $(@D)
	touch $@
