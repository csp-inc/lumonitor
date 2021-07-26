import argparse 
import os
import time

import ee
import gcsfs
from osgeo import gdal, gdalconst
import rioxarray
import xarray

def irrigated_areas(output_file):
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

    irrigated_areas(output_file=args.output_file)
