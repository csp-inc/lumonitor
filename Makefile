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

# I really am not sure why this works, some interaction between --delimiter and
# -o tsv makes it list just the paths. Alternatively use jq & json output, I guess.
BLOBS=$(shell az storage blob list --prefix 'zarr/2016/' --delimiter 'zarr' -c $(CONTAINER) --account-key=$(ACCOUNT_KEY) --account-name=$(ACCOUNT_NAME) -o tsv | sed 's/.zarr\//.zarr/' | tr '\n' ' ')

#$(info $(BLOBS))

#GCS=$(shell gsutil ls gs://lumonitor/hls_tiles/)

#GCS_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(notdir $(GCS)))
#LABEL_COGS=$(patsubst %.blob,%, $(GCS_PLACEHOLDERS))

BLOB_PLACEHOLDERS=$(patsubst %,data/blobs/%.blob, $(BLOBS))
FEATURE_COG_PLACEHOLDERS=$(patsubst data/blobs/%.zarr.blob,data/blobs/%.tif.blob, $(BLOB_PLACEHOLDERS))
$(info $(FEATURE_COG_PLACEHOLDERS))

.PHONY: all cogs predictions tiles
all: cogs
cogs: $(FEATURE_COG_PLACEHOLDERS)

$(BLOB_PLACEHOLDERS): %:
$(FEATURE_COG_PLACEHOLDERS): %.tif.blob: | %.zarr.blob

#data/cog/%.tif: data/blobs/%.tif.gcs

data/blobs/%.tif.blob: data/blobs/%.zarr.blob
	echo python3 src/utils/zarr_to_cog.py az://$(CONTAINER)/$*.zarr /vsiaz/hls/cog/2016/training/$(*F).tif --account-name=$(ACCOUNT_NAME) --account-key=$(ACCOUNT_KEY)
	echo touch $@

#data/%.gcs:
#	echo touch $@

data/%.zarr.blob: 
	echo touch $@
