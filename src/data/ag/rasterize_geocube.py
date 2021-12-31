from functools import partial
from typing import Optional, List, Union

import xarray as xr
import geopandas as gpd
import numpy as np
import rioxarray as rx
import geocube
from geocube.api.core import make_geocube
from dask.distributed import Client, Lock

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


if __name__ == '__main__':
    client = Client()
    print(client.dashboard_link)
    nodata_val = np.nan

    irr = gpd.read_file('data/irrigation_rate.gpkg')
    template = rx.open_rasterio(
        'data/irrigated_areas.tif',
        chunks={'x': 4096, 'y': 2048},
        lock=False
    )

    out_xarr = make_geocube_like_dask2(
        df=irr,
        measurements=['acre_feet_per_acre_irrigated'],
        like=template,
        fill=nodata_val
    )

    out_xarr.rio.to_raster(
        'test.tif',
        compress='LZW',
        predictor=3,
        tiled=True,
        lock=Lock("rio", client=client),
        dtype=np.float32,
        nodata=nodata_val

    )