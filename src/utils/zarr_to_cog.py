#!/usr/bin/env python3

import argparse
import os

import fsspec
import rioxarray
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('zarr')
parser.add_argument('cog')
parser.add_argument('--account-name')
parser.add_argument('--account-key')

args = parser.parse_args()

input_mapper = fsspec.get_mapper(
    args.zarr,
    account_name=args.account_name,
    account_key=args.account_key
    )

os.environ['AZURE_STORAGE_ACCOUNT'] = args.account_name
os.environ['AZURE_STORAGE_ACCESS_KEY'] = args.account_key
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

xr.open_zarr(input_mapper, mask_and_scale=False).isel(year=0).rio.to_raster(
        args.cog,
        dtype='int16',
        compress='LZW',
        predictor=2,
        tiled=True
    )
