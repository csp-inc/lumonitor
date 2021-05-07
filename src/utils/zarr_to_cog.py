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

# Needed for to_raster
os.environ['AZURE_STORAGE_ACCOUNT'] = args.account_name
os.environ['AZURE_STORAGE_ACCESS_KEY'] = args.account_key
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'


NLCD_WKT = """PROJCRS["Albers_Conical_Equal_Area",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4326]],
    CONVERSION["unnamed",
        METHOD["Albers Equal Area",
            ID["EPSG",9822]],
        PARAMETER["Latitude of false origin",23,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8821]],
        PARAMETER["Longitude of false origin",-96,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8822]],
        PARAMETER["Latitude of 1st standard parallel",29.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Latitude of 2nd standard parallel",45.5,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8824]],
        PARAMETER["Easting at false origin",0,
            LENGTHUNIT["meters",1],
            ID["EPSG",8826]],
        PARAMETER["Northing at false origin",0,
            LENGTHUNIT["meters",1],
            ID["EPSG",8827]]],
    CS[Cartesian,2],
        AXIS["easting",east,
            ORDER[1],
            LENGTHUNIT["meters",1]],
        AXIS["northing",north,
            ORDER[2],
            LENGTHUNIT["meters",1]]]"""

def convert(input_zarr, output_cog):
    input_zarr = 'az://' + input_zarr
    output_cog = '/vsiaz/' + output_cog

    print(input_zarr, output_cog)

    input_mapper = fsspec.get_mapper(
        input_zarr,
        account_name=os.environ['AZURE_STORAGE_ACCOUNT'],
        account_key=os.environ['AZURE_STORAGE_ACCESS_KEY']
    )

    (
        xr.open_zarr(input_mapper, mask_and_scale=False)
        .isel(year=0)
        .rio.reproject(
            dst_crs=NLCD_WKT,
            resolution=30)
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
    return f'{container}/cog/{year}/training_albers/{cog_file}'


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

[convert(get_zarr_for_cog(cog), cog) for cog in cogs_to_run]
