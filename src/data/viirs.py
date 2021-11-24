from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array
import rasterio as rio


if __name__ == "__main__":
    # Files had to be manually downloaded as they were behind a pw wall
    for p in Path("data/viirs").iterdir():
        # p is of the form
        # VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.tif.gz,
        year = p.name[11:15]
        template = f"data/azml/conus_hls_median_{year}.vrt"
        cutline_layer = "data/azml/conus.geojson"
        output_file = f"/vsiaz/hls/viirs_{year}.tif"
        input_ds = gdal.Open("/vsigzip/" + str(p))
        with rio.open(template) as t:
            bounds = list(t.bounds)
            crs = t.crs

        # Crop the dataset to CONUS so calculations can be done in memory
        cropped_ds = gdal.Warp(
            "",
            input_ds,
            format="VRT",
            cutlineDSName=cutline_layer,
            cropToCutline=True,
        )

        # Normalize values and convert to integer for smaller disk size
        values = cropped_ds.ReadAsArray()
        # Some very small negatives
        values[values < 0] = 0
        # Dave: sqrt of max normalized values. Here we use 5k as a "theoretical"
        # max among all years (given max among 2013, 2016, and 2020 as ~3k).
        # Then multiply by 10000 to save as integer
        values = np.sqrt(values / 5000.0) * 10000.0
        output_ds = gdal_array.OpenArray(np.int16(values))
        output_band = output_ds.GetRasterBand(1)
        output_band.SetScale(0.0001)
        gdal_array.CopyDatasetInfo(cropped_ds, output_ds)

        # Reproject to output projection and re-crop, since conversion to/from
        # ndarray lost that info
        gdal.Warp(
            output_file,
            output_ds,
            cutlineDSName="data/azml/conus_projected.gpkg",
            cropToCutline=True,
            outputBounds=bounds,
            dstNodata=-32768,
            dstSRS=crs,
            creationOptions=[
                "COMPRESS=LZW",
                "PREDICTOR=2",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "INTERLEAVE=BAND",
                "TILED=YES",
            ],
        )
