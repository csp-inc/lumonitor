import os

import numpy as np
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset

training_file = 'data/cog_mosaic.vrt'
label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'

LABEL_BAND = 1
CHIP_SIZE = 512
OUTPUT_CHIP_SIZE = 512

ds = Dataset(
    training_file,
    label_file,
    label_band=LABEL_BAND,
    feature_chip_size=CHIP_SIZE,
    label_chip_size=OUTPUT_CHIP_SIZE,
)

def worker_init_fn(worker_id):
    # ONLY Ensures different workers get different x's and y's
    np.random.seed(np.random.get_state()[1][0] + worker_id)

loader = DataLoader(
    ds,
    num_workers=1,
    batch_size=1,
    worker_init_fn=worker_init_fn,
)

for epoch in range(1, 3):
    print('epock')
    for i, data in enumerate(loader):
        print(i)
