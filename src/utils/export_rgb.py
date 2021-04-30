import argparse

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import rasterio as rio


parser = argparse.ArgumentParser()
parser.add_argument('--input-file')
parser.add_argument('--output-file')

args = parser.parse_args()

ramp = [
    (0, '#010101'),
    (1/3., '#ff0101'),
    (2/3., '#ffbb01'),
    (1, '#ffff01')
]

cm = LinearSegmentedColormap.from_list("urban", ramp)
with rio.open(args.input_file) as src:
    image_np = src.read().squeeze(0)
    kwargs = src.meta.copy()


rgb = cm(image_np, bytes=True)
# cm exports h,w,c we need c,h,w
rgb = np.moveaxis(rgb, 2, 0)

kwargs.update({
    'count': 4,
    'nodata': 0,
    'dtype': 'uint8'
})

with rio.open(args.output_file, mode='w', **kwargs) as dst:
    dst.write(rgb)

