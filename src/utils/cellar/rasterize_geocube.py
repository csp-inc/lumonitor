import argparse
from functools import partial
from typing import Callable, Optional, List, Union

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
        rasterize_function: Callable = partial(geocube.rasterize.rasterize_image, all_touched=True),
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

    # This is not setup to handle multiple measurements
    # (nor something not named 'band'?)
    like = like.rename(dict(zip(['band'], measurements)))
    return like.map_blocks(
        rasterize_block,
        template=like
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vector_file',
        help="Path containing vector data to convert to raster"
    )
    parser.add_argument(
        '--measurement_field',
        help="Field containing the value to rasterize"
    )
    parser.add_argument(
        '--template_raster',
        help="Path to raster to use as the template"
    )
    parser.add_argument(
        '--output_file',
        help="Path to the output file"
    )
    args = parser.parse_args()

    client = Client()
    print(client.dashboard_link)
    nodata_val = np.nan

    irr = gpd.read_file(args.vector_file)
    template = rx.open_rasterio(
        args.template_raster,
        chunks={'x': 4096, 'y': 2048},
        lock=False
    )

    out_xarr = make_geocube_like_dask2(
        df=irr,
        measurements=[args.measurement_field],
        like=template,
        fill=nodata_val
    )

    out_xarr.rio.to_raster(
        args.output_file,
        compress='LZW',
        predictor=3,
        tiled=True,
        lock=Lock("rio", client=client),
        dtype=np.float32,
        nodata=nodata_val
    )
