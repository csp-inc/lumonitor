import dask.array as da
import pandas as pd
import xarray as xr
from dask.distributed import Client

import hls_tooling.utils.hls as hls

def create_multiband_dataset(row, bands, chunks):
    '''A function to load multiple bands into an xarray dataset '''
    
    # Each image is a dataset containing both band4 and band5
    datasets = []
    for band, url in zip(bands, hls.scene_to_urls(row['scene'], row['sensor'], bands)):
    # needs something like export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
        da = xr.open_rasterio(url, chunks=chunks)
        da = da.squeeze().drop(labels='band')
        ds = da.to_dataset(name=band)
        datasets.append(ds)
    return xr.merge(datasets)

def create_timeseries_multiband_dataset(df, bands, chunks):
    '''For a single HLS tile create a multi-date, multi-band xarray dataset'''
    datasets = []
    for i,row in df.iterrows():
        try:
            # print('loading...', row['dt'])
            ds = create_multiband_dataset(row, bands, chunks)
            datasets.append(ds)
        except Exception as e:
            print('ERROR loading, skipping acquistion!')
            print(e)
    DS = xr.concat(datasets, dim=pd.DatetimeIndex(df['dt'].tolist(), name='time'))
    print('Dataset size (Gb): ', DS.nbytes/1e9)
    return DS

def get_mask(qa_band):
    """Takes a data array HLS qa band and returns a mask of True where quality is good, False elsewhere
    Mask usage:
        ds.where(mask)
        
    Example:
        qa_mask = get_mask(dataset[HLSBand.QA])
        ds = dataset.drop_vars(HLSBand.QA)
        masked = ds.where(qa_mask)
    """
    def is_bad_quality(qa):
        cirrus = 0b1
        cloud = 0b10
        adjacent_cloud = 0b100
        cloud_shadow = 0b1000
        high_aerosol = 0b11000000

        return (qa & cirrus > 0) | (qa & cloud > 0) | (qa & adjacent_cloud > 0) | \
            (qa & cloud_shadow > 0) | (qa & high_aerosol == high_aerosol)
    return xr.where(is_bad_quality(qa_band), False, True)  # True where is_bad_quality is False, False where is_bad_quality is True

if __name__ == '__main__':
    client = Client()
    lookup = hls.HLSTileLookup()
    tiles = list(lookup.get_point_hls_tile_ids(35, -111))
    years = [ 2016 ]
    bands = [ hls.HLSBand.COASTAL_AEROSOL, hls.HLSBand.BLUE, hls.HLSBand.GREEN,
              hls.HLSBand.QA ]
    
    cat = hls.HLSCatalog.from_tiles(list(tiles), [2016], [ "01", "02", "03"], lookup)
    chunks = { 'band':1, 'x': 366*2, 'y': 366*2 }
    tile_ds = create_timeseries_multiband_dataset(cat.xr_ds.to_dataframe(), bands, chunks)
