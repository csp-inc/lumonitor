#!/usr/bin/env python3

import argparse
import os

import fsspec
import rioxarray
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--account-name')
parser.add_argument('--account-key')

args = parser.parse_args()

os.environ['AZURE_STORAGE_ACCOUNT'] = args.account_name
os.environ['AZURE_STORAGE_ACCESS_KEY'] = args.account_key
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'


def convert(input_zarr, output_cog):
    input_mapper = fsspec.get_mapper(
        input_zarr,
        account_name=os.environ['AZURE_STORAGE_ACCOUNT'],
        account_key=os.environ['AZURE_STORAGE_ACCESS_KEY']
    )

    (
        xr.open_zarr(input_mapper, mask_and_scale=False)
        .isel(year=0)
        .rio.to_raster(
            output_cog,
            dtype='int16',
            compress='LZW',
            predictor=2,
            tiled=True
        )
    )


def get_zarr_paths(container, fs):
    walker = fs.walk(f'{container}/zarr', maxdepth=2)
    next(walker)
    return set([
        [
            os.path.join(year_dir, zarr)
            for zarr in zarrs
        ]
        for year_dir, zarrs, _ in walker
    ][0])


def get_cog_for_zarr(zarr):
    container, _, year, file = zarr.split('/')
    cog_file = os.path.splitext(file)[0] + '.tif'
    return f'{container}/cog/{year}/training/{cog_file}'


def get_cogs_for_zarrs(zarrs):
    return set([get_cog_for_zarr(z) for z in zarrs])


def get_zarr_for_cog(cog):
    container, _, year, _, file = cog.split('/')
    zarr_file = os.path.splitext(file)[0] + '.zarr'
    return f'{container}/zarr/{year}/{zarr_file}'


fs = fsspec.filesystem(
        'az',
        account_name=args.account_name,
        account_key=args.account_key
    )


input_zarrs = get_zarr_paths('hls', fs)
cogs_for_zarrs = get_cogs_for_zarrs(input_zarrs)
existing_cogs = set(fs.find('hls/cog'))

cogs_to_run = cogs_for_zarrs - existing_cogs

for cog in cogs_to_run:
    convert(get_zarr_for_cog(cog), cog)
