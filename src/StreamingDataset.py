import numpy as np

import xarray as xr
from torch.utils.data.dataset import IterableDataset

class StreamingDataset(IterableDataset):

    def __init__(self, imagery_files, label_files, label_band, chip_size=256, num_chips_per_tile=200):
        self.files = list(zip(imagery_files, label_files))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

    def stream_tiles(self):
        for file in self.files:
            yield(file[0], file[1])


    def stream_chips(self):

        for imagery_file, label_file in self.files:
            for _ in range(self.num_chips_per_tile):
                img_ds = xr.open_zarr(imagery_file)
                label_ds = xr.open_rasterio(label_file)

                x = np.random.randint(0, img_ds.dims['x'] - self.chip_size - 1)
                y = np.random.randint(0, img_ds.dims['y'] - self.chip_size - 1)
                x_cells = range(x, x + self.chip_size)
                y_cells = range(y, y + self.chip_size)

                cells = dict(x=x_cells, y=y_cells)
                img_chip = img_ds[cells].to_array().values.squeeze()
                label_chip = label_ds[cells].sel(band=1).values

                yield img_chip, label_chip

    def __iter__(self):
        return iter(self.stream_chips())
