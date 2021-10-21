import os

import azfs
import numpy as np
from osgeo import gdal, gdalconst, gdal_array
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
def convert_file(f: str, output_proj: str) -> None:
    print(f)
    nodata_val = -32768
    in_ds = gdal.Open(f"data/hm/{f}")
    big_vals = in_ds.ReadAsArray() * 10000
    np.nan_to_num(big_vals, copy=False, nan=nodata_val)
    out_ds = gdal_array.OpenArray(np.int16(big_vals))
    gdal_array.CopyDatasetInfo(in_ds, out_ds)
    gdal.Warp(
        f"data/hm_fixed/{f}",
        out_ds,
        dstSRS=output_proj,
        srcNodata=nodata_val,
        dstNodata=nodata_val,
        xRes=30,
        yRes=30,
        creationOptions=[
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "INTERLEAVE=BAND",
            "TILED=YES",
        ],
    )


os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
azc = azfs.AzFileClient(credential=os.environ["AZURE_STORAGE_ACCESS_KEY"])
output_prefix = "HM_ee_2017_v014_500_30"

output_proj = 'PROJCS["Albers Conical Equal Area",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

[
    convert_file(f, output_proj)
    for f in os.listdir("data/hm")
    if f.startswith(output_prefix)
]

# Then i did
# az storage blob upload-batch --account-name lumonitoreastus2 --account-key $AZURE_STORAGE_ACCESS_KEY --destination-path hm_fixed/ -d hls -s data/hm_fixed --max-connections 10
