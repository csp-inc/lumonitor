import argparse
import os
import time

import azfs
#import ee
import gcsfs
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from osgeo import gdal, gdalconst

#ee.Initialize()
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

@retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
def convert_file(f:str, output_proj:str) -> None:
    gdal.Warp(
        f"/vsiaz/hls/{os.path.basename(f)}",
        gdal.Open(f"/vsigs/{f}"),
        dstSRS=output_proj,
        creationOptions=["COMPRESS=LZW", "PREDICTOR=3"],
    )


def export_hm(output_proj: str) -> None:
#    image = ee.Image("projects/GEE_CSP/HM/HM_ee_2017_v014_500_30")
    output_prefix = "HM_ee_2017_v014_500_30"
#    aoi = ee.FeatureCollection("projects/GEE_CSP/thirty-by-thirty/aoi_conus")
#
#    task = ee.batch.Export.image.toCloudStorage(
#        image=image,
#        bucket="lumonitor",
#        fileNamePrefix=output_prefix,
#        region=aoi.geometry(),
#        scale=image.projection().nominalScale(),
#        maxPixels=1e11,
#    )
#
#    # task.start()
#
#    while task.active():
#        time.sleep(50)
#        print("sleepy sleepy")

    fs = gcsfs.GCSFileSystem(token=os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    azc = azfs.AzFileClient(credential="3WPere+/BOHlg4btPB29F0wi8sJQACOchgrCgHrdlahtvj747h0aQ2T3HeALhfIZkfhCjVQDbPsQg+K4xVngsQ==")
    target_files = azc.ls("https://lumonitoreastus2.blob.core.windows.net/hls")
    for f in fs.ls("lumonitor"):
        if os.path.basename(f).startswith(output_prefix):
            if not os.path.basename(f) in target_files:
                convert_file(f, output_proj)
            print(f)


if __name__ == "__main__":

    output_proj = 'PROJCS["Albers Conical Equal Area",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    export_hm(output_proj=output_proj)
