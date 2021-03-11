import numpy as np

import xarray as xr
import rioxarray as rioxr
import torch
from torch.utils.data.dataset import IterableDataset

def read_tile(path, chunks):
    return rioxr.open_rasterio(path, chunks=chunks).fillna(0)

class StreamingDataset(IterableDataset):

    def __init__(
            self,
            imagery_files,
            label_files,
            label_band=6,
            feature_chip_size=256,
            label_chip_size=256,
            num_chips_per_tile=200
    ):
        chunks = 256
        self.imagery_das = [read_tile(f, chunks) for f in imagery_files]
        self.label_das = [read_tile(f, chunks) for f in label_files]

        self.label_band = label_band
        self.feature_chip_size = feature_chip_size
        self.label_chip_size = label_chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.num_tiles = len(imagery_files)
        self.num_samples = num_chips_per_tile * len(imagery_files)

    def stream_chips(self):
        for i in range(self.num_samples):
            tile_index = np.random.randint(self.num_tiles)
            img_ds = self.imagery_das[tile_index]
            label_ds = self.label_das[tile_index]
            x = np.random.randint(0, img_ds.sizes['x'] - self.feature_chip_size)
            y = np.random.randint(0, img_ds.sizes['y'] - self.feature_chip_size)
            x_cells = range(x, x + self.feature_chip_size)
            y_cells = range(y, y + self.feature_chip_size)
            cells = dict(x=x_cells, y=y_cells)
            img_chip = img_ds[cells].values

            label_x_cells = range(x, x + self.label_chip_size)
            label_y_cells = range(y, y + self.label_chip_size)
            label_cells = dict(x=label_x_cells, y=label_y_cells)
            label_chip = label_ds[label_cells].sel(band=self.label_band).values

            yield img_chip, label_chip

    def __iter__(self):
        return iter(self.stream_chips())
