import os
import random

import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset

training_file = 'data/azml/conus_hls_median_2016.vrt'
label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'
aoi = gpd.read_file('zip+http://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_state_5m.zip')
aoi = aoi[aoi.NAME == 'Vermont']


LABEL_BAND = 1

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1337)

ds = Dataset(
    feature_file=training_file,
    aoi=aoi,
    label_file=label_file,
    num_training_chips=10,
)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # ONLY Ensures different workers get different x's and y's
#    np.random.seed(np.random.get_state()[1][0] + worker_id)

for epoch in range(1, 3):
    print('epoch')
    ds = Dataset(
        feature_file=training_file,
        aoi=aoi,
        label_file=label_file,
        num_training_chips=10,
    )
    loader = DataLoader(
        ds,
        num_workers=2,
        batch_size=1,
        worker_init_fn=worker_init_fn,
        shuffle=True
    )

    for i, (idx, window) in enumerate(loader):
        print(window)
