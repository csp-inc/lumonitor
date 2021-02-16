# Load variables in .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

VPATH=src/:data/
SHELL=/usr/bin/env bash

BLOBS=$(shell az storage blob list --delimiter 'zarr' -c hls --account-key=$(AZURE_EASTUS2_STORAGE_KEY) --account-name=lumonitoreastus2 -o tsv | tr '\n' ' ')

BLOB_PLACEHOLDERS=$(patsubst %,data/cog/%.blob, $(BLOBS))
COGS=$(patsubst %.zarr.blob,%.tif, $(BLOB_PLACEHOLDERS))

.PHONY: all
all: $(COGS)

$(BLOB_PLACEHOLDERS): %:
$(COGS): %.tif: | %.zarr.blob

data/cog/%.tif: data/cog/%.zarr.blob
	python3 src/utils/zarr_to_cog.py az://hls/$*.zarr $@ --account-name=$(AZURE_EASTUS2_STORAGE_ACCOUNT) --account-key=$(AZURE_EASTUS2_STORAGE_KEY)

data/%.blob: 
	touch $@

