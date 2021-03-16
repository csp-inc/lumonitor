import numpy as np

import rasterio as rio
from rasterio.windows import Window
import torch
from torch.utils.data.dataset import IterableDataset
from torchvision.transforms.functional import center_crop

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

            img_chip = img_ds.read(range(1,8), window=window)
            label_chip = center_crop(
                torch.Tensor(label_ds.read(self.label_band, window=window)),
                self.label_chip_size
            )
            yield (
                np.nan_to_num(img_chip, False),
                np.nan_to_num(label_chip, False)
            )

    def __iter__(self):
        return iter(self.stream_chips())
