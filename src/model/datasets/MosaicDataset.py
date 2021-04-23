import math
import os

import geopandas as gpd
import numpy as np
from numpy.ma import masked_array
import rasterio as rio
from rasterio.windows import bounds, Window
from rasterio.transform import rowcol
from shapely.geometry import box
from torch.utils.data.dataset import Dataset


class MosaicDataset(Dataset):
    def __init__(
            self,
            feature_file,
            feature_chip_size=512,
            output_chip_size=70,
            aoi=None,
            label_file=None,
            label_band=1,
            mode='train',
            num_training_chips=0,
    ):
        self.feature_file = feature_file
        self.feature_chip_size = feature_chip_size
        self.output_chip_size = output_chip_size
        self.mode = mode
        self.profile, self.bounds, self.res = self._get_raster_info()
        self.aoi = self._transform_and_buffer(aoi)

        if self.mode == 'train':
            self.label_file = label_file
            self.label_band = label_band
            self.num_chips = num_training_chips
            self.row_offs, self.col_offs = self._get_random_indices()
        else:
            self.num_chip_rows, self.num_chip_cols, self.num_chips = self._get_dims()

    def _get_dims(self):
        with rio.open(self.feature_file) as r:
            # This calc was determined by testing, so may need revisiting
            # (but appears to work)
            num_rows = 2 + (r.height - self.feature_chip_size) // self.output_chip_size
            num_cols = 2 + (r.width - self.feature_chip_size) // self.output_chip_size
        return num_rows, num_cols, num_rows * num_cols

    def _get_raster_info(self):
        with rio.open(self.feature_file) as r:
            return(r.profile, r.bounds, r.res)

    def _transform_and_buffer(self, aoi):
        if aoi is not None:
            if self.mode == 'train':
                buf = math.ceil(-1 *
                                (self.feature_chip_size / 2) *
                                self.res[0])
            else:
                # This over-estimates but who cares
                buf = self.feature_chip_size

            crs = self.profile['crs']
            buffed_gds = aoi.to_crs(crs).buffer(buf)
            return gpd.GeoDataFrame(geometry=buffed_gds)

        else:
            return None

    def _get_random_indices(self):
        x_min, y_min, x_max, y_max = self.bounds
        transform = self.profile['transform']

        # Or whatever
        n_points_to_try = self.num_chips * 5
        x = np.random.uniform(x_min, x_max, n_points_to_try)
        y = np.random.uniform(y_min, y_max, n_points_to_try)
        gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
        if self.aoi is not None:
            good_points = gdf_points.within(self.aoi.unary_union)
            gdf_points = gdf_points[good_points][0:self.num_chips]

        return rowcol(transform, gdf_points.x, gdf_points.y)

    def _get_window(self, idx):
        if self.mode == 'train':
            col_off = self.col_offs[idx]
            row_off = self.row_offs[idx]
        else:
            col_off = (idx % self.num_chip_cols) * self.output_chip_size
            max_col_off = self.profile['width'] - self.feature_chip_size
            col_off = min(col_off, max_col_off)

            row_off = (idx // self.num_chip_cols) * self.output_chip_size
            max_row_off = self.profile['height'] - self.feature_chip_size
            row_off = min(row_off, max_row_off)

        return Window(
            col_off,
            row_off,
            self.feature_chip_size,
            self.feature_chip_size
        )

    def _center_crop_window(self, window, size):
        col_off = window.col_off + window.width // 2 - (size // 2)
        row_off = window.row_off + window.height // 2 - (size // 2)
        return Window(col_off, row_off, size, size)

    def get_cropped_window(self, idx, size):
        return self._center_crop_window(self._get_window(idx), size)

    def _window_overlaps_aoi(self, window):
        if self.aoi is not None:
            window_bounds = bounds(window, self.profile['transform'])
            window_gpdf = gpd.GeoDataFrame(
                {'geometry': [box(*window_bounds)]},
                crs=self.profile['crs']
            )
            overlap = gpd.overlay(window_gpdf, self.aoi)
            return overlap.shape[0] > 0

        return True

    def _get_img_chip(self, window):
        with rio.open(self.feature_file) as img_ds:
            img_chip = img_ds.read(
                range(1, 8),
                window=window,
                masked=True
            ).filled(0) * img_ds.scales[0]
            # A couple leaps here ^^^^^
        return img_chip

    def _get_label_chip(self, window):
        label_window = self._center_crop_window(
            window,
            self.output_chip_size
        )

        with rio.open(self.label_file) as label_ds:
            label_chip_raw = label_ds.read(
                self.label_band,
                window=label_window,
            )

        imp_nodata_val = 127
        return masked_array(
            label_chip_raw,
            mask=label_chip_raw == imp_nodata_val
        ).filled(0) / 100
        # Replace nodatas with 0,
        # then divide by 100 for real values

    def __getitem__(self, idx):
        window = self._get_window(idx)

        if self._window_overlaps_aoi(window):
            img_chip = self._get_img_chip(window)

            if self.mode == 'train':
                label_chip = self._get_label_chip(window)
                return (img_chip, label_chip)

            return img_chip

        return []

    def __len__(self):
        return self.num_chips
