import argparse
import os

import azfs
from numpy import nan
from osgeo import gdal, gdalconst


def download_file(
    account_name: str,
    account_key: str,
    container_name: str,
    blob_path_prefix: str,
    local_path: str,
) -> None:
    gdal.SetConfigOption("AZURE_STORAGE_ACCOUNT", account_name)
    gdal.SetConfigOption("AZURE_STORAGE_ACCESS_KEY", account_key)

    fs = azfs.AzFileClient(credential=account_key)
    ls_path = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path_prefix}"

    pieces = [f"/vsiaz/{container_name}/{blob_path_prefix}/{f}" for f in fs.ls(ls_path)]

    print("pieces")

    vrt_tempfile = "temp.vrt"
    gdal.BuildVRT(vrt_tempfile, pieces)
    print("vrt")


#    ds = gdal.Open(vrt_tempfile)
#    ds = gdal.Translate(
#        local_path,
#        ds,
#        noData=nan,
#        creationOptions=["COMPRESS=LZW", "PREDICTOR=3"],
#    )
#    ds = None


if __name__ == "__main__":
    download_file(
        account_name="lumonitorwesteurope",
        account_key=os.environ["AZURE_LUMONITORWESTEUROPE_STORAGE_KEY"],
        container_name="lumonitorwesteurope",
        blob_path_prefix="irrigated_areas_pieces3",
        local_path="irrigated_areas_test.tif",
    )
#    parser = argparse.ArgumentParser()
#
#    parser.add_argument("output_file")
#    args = parser.parse_args()
