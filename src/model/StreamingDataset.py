import numpy as np

import xarray as xr
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
        self.files = list(zip(imagery_files, label_files))

        self.label_band = label_band
        self.feature_chip_size = feature_chip_size
        self.label_chip_size = label_chip_size
        self.num_chips_per_tile = num_chips_per_tile

    def stream_tiles(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if worker_id == 0:
            np.random.shuffle(self.files)

# This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.files)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):
            img_fn, label_fn = self.files[idx]
            yield (img_fn, label_fn)

    def stream_chips(self):

        for imagery_file, label_file in self.stream_tiles():

            img_ds = xr.open_rasterio(imagery_file).fillna(0)
            label_ds = xr.open_rasterio(label_file).fillna(0)

            for _ in range(self.num_chips_per_tile):
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
