from typing import Callable

from blend_modes import lighten_only
import dask
from dask.distributed import Client, Lock
import numpy as np
import xarray as xr
import rioxarray


def blend(da: xr.DataArray) -> xr.DataArray:
    fg = da[0:4, :, :]
    bg = da[4:8, :, :]
    blended = np.moveaxis(
        lighten_only(
            np.moveaxis(fg.to_numpy(), 0, -1),
            np.moveaxis(bg.to_numpy(), 0, -1),
            opacity=1,
        ),
        -1,
        0,
    )
    return xr.DataArray(blended, dims=fg.dims, coords=fg.coords)


if __name__ == "__main__":
    client = Client()
    print(client.dashboard_link)

    open_args = dict(chunks={"x": 8192, "y": 8192}, lock=False)
    trans = rioxarray.open_rasterio(
        "data/predictions/htrans_rgb.tif", **open_args
    ).astype("float")
    urban = rioxarray.open_rasterio(
        "data/predictions/hurban_rgb.tif", **open_args
    ).astype("float")

    # da = xr.concat([trans, urban], dim="band").chunk(dict(band=8, x=2048, y=2048))

    # da.map_blocks(blend, template=trans)
    np.maximum(trans, urban).rio.to_raster(
        "data/trans_urban.tif",
        compress="LZW",
        predictor=2,
        tiled=True,
        lock=Lock("rio", client=client),
    )
