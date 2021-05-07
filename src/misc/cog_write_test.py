import xarray as xr
import rioxarray

# f = 'data/cog/2016/training/11SKB.tif'
f = '/home/csp/temp/HLS.S30.T16TDL.2019206.v1.4_01.tif'

j = xr.open_rasterio(f)

i = 1
while True:
    j.rio.to_raster(
        '/vsiaz/lumonitor/test3.tif',
        dtype='int16',
        compress='LZW',
        predictor=2,
        tiled=True
    )
    print(i)
    i = i + 1
