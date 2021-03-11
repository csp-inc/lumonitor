import numpy as np

import xarray as xr
import rioxarray as rioxr
import torch
from torch.utils.data.dataset import Dataset

class LuDataset(Dataset):
    def __init__(
            self,
            imagery_files,
            label_files,
            label_band=6,
            feature_chip_size=256,
            label_chip_size=256,
            num_chips_per_tile=200
    ):
        self.imagery_das = [self.read_tile(f) for f in imagery_files]
        self.label_das = [self.read_tile(f) for f in label_files]

        self.label_band = label_band
        self.feature_chip_size = feature_chip_size
        self.label_chip_size = label_chip_size
        self.num_chips_per_tile = num_chips_per_tile

        self.num_chips = num_chips_per_tile * len(imagery_files)

    def __getitem__(self, idx):
        tile_idx = int(idx / self.num_chips_per_tile)

        img_ds = self.imagery_das[tile_idx]
        x = np.random.randint(0, img_ds.sizes['x'] - self.feature_chip_size)
        y = np.random.randint(0, img_ds.sizes['y'] - self.feature_chip_size)

        img_x_cells = range(x, x + self.feature_chip_size)
        img_y_cells = range(y, y + self.feature_chip_size)
        cells = dict(x=img_x_cells, y=img_y_cells)
        img_chip = img_ds[cells].values

        label_ds = self.imagery_das[tile_idx]
        label_x_cells = range(x, x + self.label_chip_size)
        label_y_cells = range(y, y + self.label_chip_size)
        label_cells = dict(x=label_x_cells, y=label_y_cells)
        label_chip = label_ds[label_cells].sel(band=self.label_band).values

        return img_chip, label_chip

    def __len__(self):
        return self.num_chips

    def read_tile(self, path, chunks=256):
        return rioxr.open_rasterio(path, chunks=(-1, 256, 256), lock=False).fillna(0)
