import os

import ee
import geopandas as gpd

ee.Initialize()

from HLSTileLookup import HLSTileLookup

def export_tile_for_ee_image(ee_image, xmin, ymax, epsg, tile_id, prefix):
    res = 30
    cells = 3600
    diff = res * cells
    xmax = xmin + diff
    ymin = ymax - diff
    epsg_string = 'EPSG:' + str(epsg)
    projection = ee.Projection(epsg_string)
    tile = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], projection, False)

    task = ee.batch.Export.image.toCloudStorage(
        image=ee_image,
        bucket='lumonitor',
        fileNamePrefix=os.path.join('hls_tiles', prefix + '_' + tile_id),
        dimensions=cells,
        region=tile,
        crs=epsg_string
        )
    task.start()


def export_tiles_for_ee_image(ee_image, aoi):
    tile_lookup = HLSTileLookup()
    tile_df = tile_lookup.get_geometry_hls_tile_info(aoi)
    for _, row in tile_df.iterrows():
        export_tile_for_ee_image(
            ee_image=ee_image,
            xmin=row['Xstart'],
            ymax=row['Ystart'],
            epsg=row['EPSG'],
            tile_id=row['TilID'],
            prefix='hm'
        )

# Yes i know i will fix this
aoi = gpd.read_file('/home/csp/Projects/thirty-by-thirty/data/aoi_conus.geojson')

hm = ee.Image('projects/GEE_CSP/HM/HM_ee_2017_v014_500_30')
nlcd_imp = ee.Image('USGS/NLCD/NLCD2016').select('impervious').divide(100).float()
hm = hm.addBands(nlcd_imp)

bbox = [-124.9, 24.4, -66.7, 49.5]
export_tiles_for_ee_image(hm, aoi)
