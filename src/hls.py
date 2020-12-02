from azure.storage.blob import ContainerClient
import geopandas as gp
from osgeo import gdal
import numpy as np
import numpy.ma as ma
import os

def get_summaries(tile_id, year):
    # I do this for multiple sdss so we don't have to re-get QA band.
    # There may be better ways...
    all_paths = get_paths(tile_id, year)

    sdss = [ '01','02','03','04','05','06','07','08','09','10'] 
    stacks = dict.fromkeys(sdss, [])
    for daynum in get_daynums(all_paths):
        qa_path = filter_daynum(filter_subdatasets(all_paths, '11'), daynum)[0]
        qa_ds = gdal.Open(qa_path)
        qa = np.array(qa_ds.GetRasterBand(1).ReadAsArray())
        mask = get_mask(qa)
        for sds in sdss:
            img_path = filter_daynum(filter_subdatasets(all_paths,sds), daynum)[0]
            print(img_path)
            img_ds = gdal.Open(img_path)
            img = np.array(img_ds.GetRasterBand(1).ReadAsArray()).astype(np.float64)
            img[mask] = np.nan
            stacks[sds].append(img)

    summaries = dict(summaries=dict())
    for sds in sdss:
        summaries['summaries'][sds] = np.nanmedian(np.array(stacks[sds]), 0)

    # Assume profile is the same for all images in tile, may check
    some_path = filter_subdatasets(all_paths, '01')[0]
    summaries['model_ds'] = gdal.Open(some_path)

    return summaries
        
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

    prefix = 'L309/HLS.L30.T' + tile_id + '.' + str(year)
    # needs to be streaming or it downloads the whole thing
    vfs = '/vsiaz_streaming/hls/'
    return [ vfs + blob.name for blob in cc.list_blobs(name_starts_with=prefix) ]

def get_daynums(paths):
    if not isinstance(paths, list):
        paths = [ paths ] 
    # Not sure this is the "safest" way to get the string
    daynums = list(set([ os.path.basename(p)[19:22] for p in paths]))
    if len(daynums) == 1:
        return daynums[0]
    else:
        return daynums

def filter_daynum(paths, daynum):
    return [p for p in paths if get_daynums(p) == daynum ]

def filter_subdatasets(paths, subdatasets):
    if not isinstance(subdatasets, list):
        subdatasets = [subdatasets]

    suffixes = tuple([ s + '.tif' for s in subdatasets ])
    return [ p for p in paths if p.endswith(suffixes) ]

def get_mask(qa):
    cirrus = 0b1
    cloud = 0b10
    adjacent_cloud = 0b100
    cloud_shadow = 0b1000
    high_aerosol = 0b11000000

    return (qa & cirrus > 0) | (qa & cloud > 0) | (qa & adjacent_cloud > 0) | \
        (qa & cloud_shadow > 0) | (qa & high_aerosol == high_aerosol)

def create_vrt(paths, output_file):
    # GDAL 3.2 needed for options
    opts = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(output_file, paths, options=opts)

def write_array_to_tiff(array, output_path, model_ds):
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.CreateCopy(output_path, model_ds, strict=0)
    output_ds.GetRasterBand(1).WriteArray(array)

    model_ds = None
    output_ds = None


tile_id = '16TDL'
year = '2016'
#output_dir = 'data/lumonitor-eastus2'
output_dir = 'data'

gdal.SetConfigOption("AZURE_SAS", "st=2019-08-07T14%3A54%3A43Z&se=2050-08-08T14%3A54%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=EYNJCexDl5yxb1TxNH%2FzILznc3TiAnJq%2FPvCumkuV5U%3D")
gdal.SetConfigOption("AZURE_STORAGE_ACCOUNT", "hlssa")
gdal.SetConfigOption("GTIFF_IGNORE_READ_ERRORS", "YES")

summaries = get_summaries(tile_id, year)
for (sds, summary) in summaries['summaries'].items():
    output_file = os.path.join(output_dir, 
            tile_id + '_' + year + '_' + sds + '.tif')

    write_array_to_tiff(summary, output_file, summaries['model_ds'])
