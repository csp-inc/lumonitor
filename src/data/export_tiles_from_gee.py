import argparse
import os

import ee
import rasterio

ee.Initialize()

parser = argparse.ArgumentParser()
parser.add_argument('cog')
args = parser.parse_args()
cog = rasterio.open(args.cog)


tile_id = os.path.splitext(os.path.basename(args.cog))[0]
cells = cog.profile['width']

bounds = list(cog.bounds)
epsg_string = f'EPSG:{cog.crs.to_epsg()}'
projection = ee.Projection(epsg_string)
tile = ee.Geometry.Rectangle(bounds, projection, False)

hm = ee.Image('projects/GEE_CSP/HM/HM_ee_2017_v014_500_30')
nlcd_imp = ee.Image('USGS/NLCD/NLCD2016').select('impervious').divide(100).float()
hm = hm.addBands(nlcd_imp)

task = ee.batch.Export.image.toCloudStorage(
    image=hm,
    bucket='lumonitor',
    fileNamePrefix=os.path.join('hls_tiles', f'hm_{tile_id}'),
    dimensions=cells,
    region=tile,
    crs=epsg_string
    )

task.start()
