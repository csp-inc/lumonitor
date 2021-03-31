import os

import numpy as np
from torch.utils.data import DataLoader

#from SerialStreamingDataset import SerialStreamingDataset
from LuDataset import LuDataset as Dataset

cog_dir = 'data/cog/2016/training'
image_files = [
    os.path.join(cog_dir, f)
    for f in os.listdir(cog_dir)
    if not f.startswith('hm')
][0:4]

training_files = image_files
tiles = [os.path.splitext(os.path.basename(f))[0] for f in training_files]
label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in tiles]


LABEL_BAND = 6
CHIP_SIZE = 512
OUTPUT_CHIP_SIZE = 512
N_CHIPS_PER_TILE = 5

ds = Dataset(
    training_files,
    label_files,
    label_band=LABEL_BAND,
    feature_chip_size=CHIP_SIZE,
    label_chip_size=OUTPUT_CHIP_SIZE,
    num_chips_per_tile=N_CHIPS_PER_TILE
)

def worker_init_fn(worker_id):
    # ONLY Ensures different workers get different x's and y's
    np.random.seed(np.random.get_state()[1][0] + worker_id)

loader = DataLoader(
    ds,
    num_workers=2,
    batch_size=1,
    worker_init_fn=worker_init_fn,
    # Shuffles file indexes, but NOT x's and y's
    shuffle=True
)

for epoch in range(1, 3):
    print('epock')
    for i, data in enumerate(loader):
        print(data)
