import os

import ee
import geopandas as gpd
import xarray as xr

ee.Initialize()

from HLSTileLookup import HLSTileLookup

def export_tile_for_ee_image(ee_image, xmin, ymax, epsg, tile_id, prefix):
    # I wonder if we should hardcore these from the tile info?
    res = 30
    cells = 3660
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

def export_tiles(ee_image, tile_info):
    for _, row in tile_info.iterrows():
        export_tile_for_ee_image(
            ee_image=ee_image,
            xmin=row['Xstart'],
            ymax=row['Ystart'],
            epsg=row['EPSG'],
            tile_id=row['TilID'],
            prefix='hm'
        )

def export_tiles_for_aoi(ee_image, aoi):
    tile_lookup = HLSTileLookup()
    tile_df = tile_lookup.get_geometry_hls_tile_info(aoi)
    export_tiles(ee_image, tile_df)

def export_tiles_for_tile_ids(ee_image, tile_ids):
    tile_lookup = HLSTileLookup()
    tile_df = tile_lookup.get_tile_info(tile_ids)
    export_tiles(ee_image, tile_df)

zarrs = [os.path.join('hls_tmp', f) for f in os.listdir('hls_tmp') if f.endswith('zarr')]
tile_ids = [xr.open_zarr(f).attrs['SENTINEL2_TILEID'] for f in zarrs]

hm = ee.Image('projects/GEE_CSP/HM/HM_ee_2017_v014_500_30')
nlcd_imp = ee.Image('USGS/NLCD/NLCD2016').select('impervious').divide(100).float()
hm = hm.addBands(nlcd_imp)

export_tiles_for_tile_ids(hm, tile_ids)

# Yes i know i will fix this
# aoi = gpd.read_file('/home/csp/Projects/thirty-by-thirty/data/aoi_conus.geojson')
# export_tiles_for_ee_image(hm, aoi)

