import argparse

import dask
import xarray
import rioxarray

from dask.distributed import Client, Lock


def is_cropland(cdl: xarray.DataArray) -> xarray.DataArray:
    """
    Reclassifies cdl into cropland T/F
    """
    return (
        ((cdl > 0) & (cdl < 61)) |
        ((cdl > 65) & (cdl < 78)) |
        ((cdl > 194) & (cdl < 256))
    )


def cropland(year: int, output_file: str) -> None:
    """
    Downloads cdl for a given year and calculates cropland/non-cropland, then
    saves
    """

    url = f'/vsizip/vsicurl/https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{year}_30m_cdls.zip/{year}_30m_cdls.img'

    cdl_2020 = rioxarray.open_rasterio(
        url,
        chunks={'x': 4096, 'y': 4096},
        lock=False,
    )

    nodata_val = 255
    client = Client()

    (
        is_cropland(cdl_2020)
        .where(cdl_2020 != cdl_2020.rio.nodata, nodata_val)
        # nodata doesn't persist if cast after set_nodata
        .astype('uint8')
        .rio.set_nodata(nodata_val, inplace=True)
        .rio.to_raster(
            output_file,
            compress='LZW',
            predictor=2,
            tiled=True,
            lock=Lock("rio", client=client)
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    args = parser.parse_args()

    cropland(year=2020, output_file=args.output_file)
