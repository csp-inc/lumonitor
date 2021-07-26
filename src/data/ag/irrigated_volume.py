import argparse

import dask
import rasterio
import rioxarray
import xarray

from dask.distributed import Client, Lock

def irrigated_volume(
    state_irrigation_volume_file: str,
    irrigated_areas_file: str,
    output_file: str
) -> None:
    # 1. acres_irrigated = irrigated_areas * pixel_size * acresperpixel
with rasterio.open(irrigated_areas_file) as src:
    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as vrt:
        irrigated_areas = rioxarray.open_rasterio(
            vrt,
            chunks={'x': 4096, 'y': 4096},
            lock=False
        )
        res = irrigated_areas.rio.resolution()
        pixel_area_sq_meters = abs(res[0] * res[1])
        acres_per_sq_meters = 2.47105e-4
        pixel_area_acres = pixel_area_sq_meters * acres_per_sq_meters
        acres_irrigated = irrigated_areas * pixel_area_acres


            





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('irrigation_rate_file')
    parser.add_argument('irrigated_areas_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    client = Client()
    print(client.dashboard_link)

    irrigated_volume(
        args.irrigation_rate_file,
        args.irrigated_areas_file,
        args.output_file
    )

