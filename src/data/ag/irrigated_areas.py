import argparse 
import os
import time

import ee
import gcsfs
from osgeo import gdal, gdalconst
import rioxarray
import xarray

def irrigated_areas(output_proj, output_file):
    ee.Initialize()

    aoi = ee.FeatureCollection("projects/GEE_CSP/thirty-by-thirty/aoi_conus")
    no_data = 255
    mask = ee.Image().paint(aoi, no_data)

    lanid = (
        ee.Image("users/xyhisu/irrigationMapping/results/LANID12")
        .byte()
        .unmask(0)
        .updateMask(mask)
        .unmask(no_data)
    )
    output_prefix = 'LANID12'

    task = ee.batch.Export.image.toCloudStorage(**{
        'image': lanid,
        'bucket': "lumonitor",
        'fileNamePrefix': output_prefix,
        'region': aoi.geometry(),
        'scale': lanid.projection().nominalScale(),
        'crs': output_proj,
        'maxPixels': 1e11,
        'fileDimensions': 131072
    })

    task.start()

    while task.active():
        time.sleep(50)
        print('Back to sleep')

    fs = gcsfs.GCSFileSystem()
    pieces = [
        f'/vsigs/{f}' for f in fs.ls('lumonitor')
        if os.path.basename(f).startswith(output_prefix)
    ]

    vrt_tempfile = f'/vsimem/{output_prefix}.vrt'
    gdal.BuildVRT(vrt_tempfile, pieces)

    ds = gdal.Open(vrt_tempfile)
    ds = gdal.Translate(
        output_file,
        ds,
        outputType=gdalconst.GDT_Byte,
        noData=no_data,
        creationOptions=['COMPRESS=LZW', 'PREDICTOR=2']
    )
    ds = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    args = parser.parse_args()

    output_proj = 'PROJCS["Albers Conical Equal Area",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    irrigated_areas(output_proj=output_proj, output_file=args.output_file)
