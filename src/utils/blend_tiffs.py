from typing import Callable

from blend_modes import lighten_only
import dask
from dask.distributed import Client, Lock
import numpy as np
import xarray as xr
import rioxarray


def blend(a, b):
    print(a.shape)
    return lighten_only(a, b, opacity=1)


if __name__ == "__main__":
    client = Client()
    print(client.dashboard_link)

    open_args = dict(chunks={"x": 4096, "y": 4096}, lock=False)
    trans = (
        rioxarray.open_rasterio("data/predictions/htrans_rgb.tif", **open_args).astype(
            "float"
        )
        #        .transpose("x", "y", "band")
    )
    urban = (
        rioxarray.open_rasterio("data/predictions/hurban_rgb.tif", **open_args).astype(
            "float"
        )
        #        .transpose("x", "y", "band")
    )

    xr.apply_ufunc(
        blend,
        trans,
        urban,
        dask="parallelized",
        input_core_dims=[["band"], ["band"]],
        exclude_dims=set(("band",)),
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    ).rio.to_raster(
        "data/trans_urban.tif",
        compress="LZW",
        predictor=2,
        tiled=True,
        lock=Lock("rio", client=client),
    )
