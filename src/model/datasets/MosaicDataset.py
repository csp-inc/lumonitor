import math
import os

import geopandas as gpd
import numpy as np
from numpy.ma import masked_array
import rasterio as rio
from rasterio.windows import Window
from rasterio.transform import rowcol
from torch.utils.data.dataset import Dataset

class MosaicDataset(Dataset):
    def __init__(
            self,
            imagery_file,
            label_file,
            label_band=1,
            feature_chip_size=256,
            label_chip_size=256,
            num_chips=1000,
            aoi=None
    ):
        self.label_band = label_band
        self.feature_chip_size = feature_chip_size
        self.label_chip_size = label_chip_size
        self.num_chips = num_chips
        self.imagery_file = imagery_file
        self.label_file = label_file
        self.aoi = self._transform_and_buffer(aoi)
        self.row_offs, self.col_offs = self._get_random_indices()

    def _transform_and_buffer(self, aoi):
        if aoi is not None:
            with rio.open(self.imagery_file) as r:
                crs = r.crs
                buf = math.ceil(-1 *
                                (self.feature_chip_size / 2) *
                                r.res[0])
            return aoi.transform(crs).buffer(buf)
        else:
            return None

    def _get_random_indices(self):
        with rio.open(self.imagery_file) as r:
            x_min, y_min, x_max, y_max = r.bounds
            transform = r.transform

        # Or whatever
        n_points_to_try = self.num_chips * 5
        x = np.random.uniform(x_min, x_max, n_points_to_try)
        y = np.random.uniform(y_min, y_max, n_points_to_try)
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        if self.aoi is not None:
            good_points = gdf_points.within(self.aoi.unary_union)
            gdf_points = gdf_points[good_points][0:self.num_chips]

        return rowcol(transform, gdf_points.x, gdf_points.y)

    def _center_crop(self, window, size):
        col_off = window.col_off + window.width // 2 - (size // 2)
        row_off = window.row_off + window.height // 2 - (size // 2)
        return Window(col_off, row_off, size, size)

    def __getitem__(self, idx):
        img_ds = rio.open(self.imagery_file)
        label_ds = rio.open(self.label_file)

        window = Window(
            # Everywhere else it's row, column. This makes some sense
            # b/c this is ultimately x,y but :exploding_head:
            self.col_offs[idx],
            self.row_offs[idx],
            self.feature_chip_size,
            self.feature_chip_size
        )
        img_chip = img_ds.read(
            range(1, 8),
            window=window,
            masked=True
        ).filled(0) * img_ds.scales[0]
        # A couple leaps here ^^^^^

        label_window = self._center_crop(window, self.label_chip_size)
        label_chip_raw = label_ds.read(self.label_band, window=label_window)
        imp_nodata_val = 127
        label_chip = masked_array(
            label_chip_raw,
            mask=label_chip_raw == imp_nodata_val
        ).filled(0) / 100
        # Replace nodatas with 0, then divide by 100 for real values

        return (img_chip, label_chip)

    def __len__(self):
        return self.num_chips
