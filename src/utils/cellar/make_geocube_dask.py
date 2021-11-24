from functools import partial
from typing import Optional, List

import xarray as xr
import geopandas as gpd
import numpy as np
import rioxarray as rx
import geocube
from geocube.api.core import make_geocube

def make_geocube_dask(
        df: gpd.GeoDataFrame,
        measurements: Optional[List[str]],
        like: xr.core.dataarray.DataArray,
        fill: int=0,
        rasterize_function:callable=partial(geocube.rasterize.rasterize_image, all_touched=True),
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
