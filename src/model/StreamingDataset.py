import numpy as np

import rasterio as rio
from rasterio.windows import Window
import torch
from torch.utils.data.dataset import IterableDataset

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
        self.imagery_files = imagery_files
        self.label_files = label_files

        self.label_band = label_band
        self.feature_chip_size = feature_chip_size
        self.label_chip_size = label_chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.num_tiles = len(imagery_files)
        self.num_samples = num_chips_per_tile * len(imagery_files)

    def __center_crop(self, window, size):
        row_off = window.height // 2 - (size // 2)
        col_off = window.width // 2 - (size // 2)
        return Window(row_off, col_off, size, size)

    def stream_chips(self):
        for i in range(self.num_samples):
            tile_index = np.random.randint(self.num_tiles)
            img_ds = rio.open(self.imagery_files[tile_index])
            label_ds = rio.open(self.label_files[tile_index])
            x = np.random.randint(0, img_ds.shape[0] - self.feature_chip_size)
            y = np.random.randint(0, img_ds.shape[1] - self.feature_chip_size)
            window = Window(
                x,
                y,
                self.feature_chip_size,
                self.feature_chip_size
            )

            img_chip = img_ds.read(range(1, 8), window=window)
            label_window = self.__center_crop(window, self.label_chip_size)
            label_chip = label_ds.read(self.label_band, window=label_window)

            yield (
                np.nan_to_num(img_chip, False),
                np.nan_to_num(label_chip, False)
            )

    def __iter__(self):
        return iter(self.stream_chips())
