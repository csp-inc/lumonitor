import numpy as np

import rasterio as rio
from rasterio.windows import Window
import torch
from torch.utils.data.dataset import IterableDataset

class SerialStreamingDataset(IterableDataset):

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

        N = len(self.files)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id+1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):
            img_fn, label_fn = self.files[idx]
            yield (img_fn, label_fn)

    def __center_crop(self, window, size):
        col_off = window.col_off + window.width // 2 - (size // 2)
        row_off = window.row_off + window.height // 2 - (size // 2)
        return Window(col_off, row_off, size, size)

    def stream_chips(self):

        for imagery_file, label_file in self.stream_tiles():
            img_ds = rio.open(imagery_file)
            label_ds = rio.open(label_file)

            for _ in range(self.num_chips_per_tile):
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
                    imagery_file,
                    x,
                    y,
#                    np.nan_to_num(img_chip, False),
#                    np.nan_to_num(label_chip, False)
                )

    def __iter__(self):
        return iter(self.stream_chips())
