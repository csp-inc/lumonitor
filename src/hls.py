from azure.storage.blob import ContainerClient
from osgeo import gdal
import geopandas as gp
# Only needed for my test data
from shapely.geometry import Polygon

def get_tile_ids(gp_df):
    tiles = gp.read_file('data/lumonitor-eastus2/mgrs_region.shp')
    overlapping_tiles = gp.overlay(gp_df, tiles, how="intersection")
    return list(overlapping_tiles['GRID1MIL'] + overlapping_tiles['GRID100K'])

def get_paths(tile_id, year):
    # For options and path structure, see https://azure.microsoft.com/en-us/services/open-datasets/catalog/hls/
    cc = ContainerClient(
            account_url="https://hlssa.blob.core.windows.net", 
            container_name='hls', 
            credential="st=2019-08-07T14%3A54%3A43Z&se=2050-08-08T14%3A54%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=EYNJCexDl5yxb1TxNH%2FzILznc3TiAnJq%2FPvCumkuV5U%3D"
            )

    paths = []
    for tile_id in tile_ids:
        prefix = 'L309/HLS.L30.T' + tile_id + '.' + str(year)
        # needs to be streaming or it downloads the whole thing
        vfs = '/vsiaz_streaming/hls/'
        these_paths = [ vfs + blob.name for blob in cc.list_blobs(name_starts_with=prefix) ]
        paths = paths + these_paths


    return paths

def create_vrt(paths, output_file):
    # GDAL 3.2 needed for options
    gdal.SetConfigOption("AZURE_SAS", "st=2019-08-07T14%3A54%3A43Z&se=2050-08-08T14%3A54%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=EYNJCexDl5yxb1TxNH%2FzILznc3TiAnJq%2FPvCumkuV5U%3D")
    gdal.SetConfigOption("AZURE_STORAGE_ACCOUNT", "hlssa")
    opts = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(output_file, paths, options=opts)

# Just for testing purposes
area = gp.GeoDataFrame({'geometry': gp.GeoSeries([Polygon([(-126,24),(-66, 24), (-126, 50), (-66,50)])]),
                        'id':[1]})
area = area.set_crs('EPSG:4326')

tile_ids = get_tile_ids(area)
paths = get_paths(tile_ids, 2016)

create_vrt(paths, 'data/lumonitor-eastus2/today.vrt')
