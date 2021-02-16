import os

from torch.utils.data import DataLoader
import xarray as xr

from StreamingDataset import StreamingDataset

cog_dir = 'data/cog/2016'
cogs = os.path.join(cog_dir, os.path.listdir(cog_dir))

label_files = [os.path.join(zarr_dir, 'hm_' + t + '.tif') for t in cogs]

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

