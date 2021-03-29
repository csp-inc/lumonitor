#!/usr/bin/env python3

import argparse
import os

import fsspec
import rasterio as rr
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

xr.open_zarr(input_mapper).isel(year=0).rio.to_raster(
        args.cog,
        dtype='float32',
        compress='LZW',
        predictor=3,
        tiled=True
    )
