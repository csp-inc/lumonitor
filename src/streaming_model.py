import os

from torch.utils.data import DataLoader
import xarray as xr

from StreamingDataset import StreamingDataset

zarr_dir = 'hls_tmp'

zarr_files = [
        os.path.join(zarr_dir, f)
        for f in os.listdir(zarr_dir)
        if f.endswith('zarr')
    ]

tiles = [xr.open_zarr(f).attrs['SENTINEL2_TILEID'] for f in zarr_files]
image_files = [os.path.join(zarr_dir, 'hls_' + t + '.tif') for t in tiles]
label_files = [os.path.join(zarr_dir, 'hm_' + t + '.tif') for t in tiles]

szd = StreamingDataset(image_files, label_files, label_band='impervious', num_chips_per_tile=10000)

loader = DataLoader(szd)

import time

start = time.time()

total_i = 0
for i, data in enumerate(loader):
    total_i = total_i + 1
    print(total_i)

end = time.time()
print(end - start)

