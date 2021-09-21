import argparse
from math import prod
from functools import partial
from typing import Optional, List, Union

import dask
from dask.distributed import Client, Lock
import geocube
from geocube.api.core import make_geocube
import geopandas as gpd
import numpy as np
import rioxarray as rx
import xarray as xr
import xrspatial

def zonal_stats(df: gpd.GeoDataFrame, zone_col: str, value_da: str, like: xr.Dataset):

    def zonal_stats_block(block):
        zones = make_geocube(
            vector_data=df,
            measurements=[zone_col],
            like=block,
            fill=np.nan,
            rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
        ).assign_coords(block.coords)
        zones[value_da] = block[value_da]
        # groupby complains if zone_col is all nan within the block
        if np.isnan(zones.id).all().item(0):
            return zones
        return zones

    return like.map_blocks(
        zonal_stats_block,
        template=like
    ).drop('spatial_ref')


def make_geocube_like_dask2(
        df: gpd.GeoDataFrame,
        measurements: Optional[List[str]],
        like: xr.core.dataarray.DataArray,
        fill: Union[int, float] = 0,
        rasterize_function: callable = partial(geocube.rasterize.rasterize_image, all_touched=True),
        **kwargs
):
    def rasterize_block(block):
        return(
            make_geocube(
                df,
                measurements=measurements,
                like=block,
                fill=fill,
                rasterize_function=rasterize_function,
            )
            .to_array(measurements[0])
            .assign_coords(block.coords)
        )

    like = like.rename(dict(zip(['band'], measurements)))
    return like.map_blocks(
        rasterize_block,
        template=like
    )


def binary_to_acres(binary: xr.DataArray) -> xr.DataArray:
    """
    Converts a 1/0 image to the acres of 1 in each cell. Cell units must be in
    square meters
    """
    sq_meters_per_cell = abs(prod(binary.rio.resolution()))
    acres_per_sq_meter = 2.47105e-6
    acres_per_cell = sq_meters_per_cell * acres_per_sq_meter
    return binary * acres_per_cell

#def acre_feet_per_cell(acre_feet_per_acres: xr.DataArray, irrigated_areas: xr.DataArray) -> xr.DataArray:
#    total_acres = binary_to_acres(irrigated_areas).sum()


if __name__ == '__main__':
    client = Client()
    print(client.dashboard_link)

    irrigated_areas = binary_to_acres(rx.open_rasterio(
#        'ag1_5070.tif',
        'data/irrigated_areas.tif',
        chunks={'x': 4096, 'y': 8192},
        masked=True,
        lock=False
    ))

#    states = gpd.read_file('states.shp').to_crs(irrigated_areas.rio.crs.to_wkt())
    states = gpd.read_file('states.shp').to_crs(irrigated_areas.rio.crs.to_wkt())
    id_col = 'STATEFP'
    states[id_col] = states[id_col].astype('int')
    states_da = make_geocube_like_dask2(
        df=states,
        measurements=[id_col],
        like=irrigated_areas,
        fill=0
    )

    zs = xrspatial.zonal_stats(states_da.squeeze(), irrigated_areas.squeeze(), ['sum'])
    print(zs.compute())

