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

BLOBS=$(shell az storage blob list --delimiter 'zarr' -c hls --account-key=$(AZURE_EASTUS2_STORAGE_KEY) --account-name=lumonitoreastus2 -o tsv | tr '\n' ' ')
$(INFO $(BLOBS))

#GCS=$(shell gsutil ls gs://lumonitor/hls_tiles/)

#GCS_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(notdir $(GCS)))
#LABEL_COGS=$(patsubst %.blob,%, $(GCS_PLACEHOLDERS))

BLOB_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(BLOBS))
FEATURE_COGS=$(patsubst data/blobs/%.zarr.blob,data/cog/%.tif, $(BLOB_PLACEHOLDERS))
$(info $(FEATURE_COGS))

.PHONY: all
all: $(FEATURE_COGS)

$(BLOB_PLACEHOLDERS): %:
$(FEATURE_COGS): %.tif: | %.zarr.blob

data/cog/%.tif: data/blobs/%.tif.gcs

data/cog/%.tif: data/blobs/%.zarr.blob
	python3 src/utils/zarr_to_cog.py az://hls/$*.zarr $@ --account-name=$(AZURE_EASTUS2_STORAGE_ACCOUNT) --account-key=$(AZURE_EASTUS2_STORAGE_KEY)

data/%.gcs:
	touch $@

data/%.blob: 
	touch $@

