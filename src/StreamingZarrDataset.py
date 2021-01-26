import numpy as np

import xarray as xr
from torch.utils.data.dataset import IterableDataset

class StreamingZarrDataset(IterableDataset):

    def __init__(self, imagery_files, chip_size=256, num_chips_per_tile=200):
        self.files = imagery_files

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

    def stream_chips(self):

        for file in self.files:
            img_ds = xr.open_zarr(file)

            x = np.random.randint(0, img_ds.dims['x'] - self.chip_size)
            y = np.random.randint(0, img_ds.dims['y'] - self.chip_size)
            x_cells = range(x, x + self.chip_size)
            y_cells = range(y, y + self.chip_size)

            chip = img_ds[dict(x=x_cells, y=y_cells)]

            yield chip.to_array().values.squeeze()

    def __iter__(self):
        return iter(self.stream_chips())









