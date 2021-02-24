import numpy as np

import xarray as xr
from torch.utils.data.dataset import IterableDataset

class StreamingDataset(IterableDataset):

    def __init__(
            self,
            imagery_files,
            label_files,
            label_band,
            chip_size=256,
            num_chips_per_tile=200
    ):
        self.files = list(zip(imagery_files, label_files))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

    def stream_tiles(self):
        for file in self.files:
            yield(file[0], file[1])

    def stream_chips(self):

        for imagery_file, label_file in self.files:

            # Not sure about best ops here for performance. 
            # chunks of 3660? just use cache?
            img_ds = xr.open_rasterio(imagery_file, cache=True).fillna(0)
            label_ds = xr.open_rasterio(label_file, cache=True).fillna(0)

            for _ in range(self.num_chips_per_tile):
                x = np.random.randint(0, img_ds.sizes['x'] - self.chip_size)
                y = np.random.randint(0, img_ds.sizes['y'] - self.chip_size)
                x_cells = range(x, x + self.chip_size)
                y_cells = range(y, y + self.chip_size)

                cells = dict(x=x_cells, y=y_cells)
                img_chip = img_ds[cells].values
                label_chip = label_ds[cells].sel(band=6).values

                yield img_chip, label_chip

    def __iter__(self):
        return iter(self.stream_chips())
