import dask
import xarray
import rioxarray

from dask.distributed import Client, Lock

def is_cropland(da):
    return ((da > 0) & (da < 61)) | ((da > 65) & (da < 78)) | ((da > 194) & (da < 256))

cdl = rioxarray.open_rasterio(
        '/home/csp/Downloads/2020_30m_cdls/2020_30m_cdls.tiff',
        chunks={'x': 4096, 'y': 4096},
        lock=False,
    )

nodata_val = 255
client = Client()

(
    is_cropland(cdl)
    .where(cdl != cdl.rio.nodata, nodata_val)
    # nodata doesn't persist if cast after set_nodata
    .astype('uint8')
    .rio.set_nodata(nodata_val, inplace=True)
    .rio.to_raster(
        'test2.tif',
        compress='LZW',
        predictor=2,
        tiled=True,
        lock=Lock("rio", client=client)
    )
)
