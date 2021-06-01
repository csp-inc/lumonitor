import argparse

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import rioxarray
#import rasterio as rio
import xarray as xr


parser = argparse.ArgumentParser()
parser.add_argument('--input-file')
parser.add_argument('--output-file')

args = parser.parse_args()

def rgbify(arr: np.ndarray) -> np.ndarray:
    print('a')
    ramp = [
        (0, '#010101'),
        (1/3., '#ff0101'),
        (2/3., '#ffbb01'),
        (1, '#ffff01')
    ]
    cm = LinearSegmentedColormap.from_list("urban", ramp)
    print(arr.shape)
    rgb = cm(arr, bytes=True)
    print('b')
    # cm exports h,w,c we need c,h,w
    return np.moveaxis(rgb, 2, 0)


j = xr.open_rasterio(
    args.input_file,
    chunks={'band': 1, 'x': 256, 'y': 256}
)

print('starting')
rgb = xr.apply_ufunc(rgbify, j, dask='allowed')
print("done")

rgb.rio.to_raster(
    args.output_file,
    dtype='int16',
    compress='LZW',
    predictor=2,
    tiled=True
)

#with rio.open(args.input_file) as src:
#    kwargs = src.meta.copy()
#
#kwargs.update({
#    'count': 4,
#    'nodata': 0,
#    'dtype': 'uint8',
#    'compress': 'LZW',
#    'predictor': 2
#})
#
#
