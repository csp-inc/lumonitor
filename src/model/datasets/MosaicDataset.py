import math
import os

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
from numpy.ma import masked_array
import rasterio as rio
from rasterio.windows import bounds, Window
from rasterio.transform import rowcol
from shapely.geometry import box, Point
from tenacity import retry, stop_after_attempt, wait_fixed
from torch.utils.data.dataset import Dataset

def range_with_end(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end

class MosaicDataset(Dataset):
    def __init__(
            self,
            feature_file: str,
            feature_chip_size: int = 512,
            output_chip_size: int = 70,
            aoi: GeoDataFrame = None,
            label_file: str = None,
            label_band: int = 1,
            mode: str = 'train',
            num_training_chips: int = 0,
    ) -> None:
        self.feature_file = feature_file
        self.feature_chip_size = feature_chip_size
        self.output_chip_size = output_chip_size
        self.mode = mode
        self.profile, self.raster_bounds, self.res = self._get_raster_info()
        self.aoi = self._transform_and_buffer(aoi)
        self.bounds = self._get_bounds()

        if self.mode == 'train':
            self.label_file = label_file
            self.label_band = label_band
            self.num_chips = num_training_chips
            self.row_offs, self.col_offs = self._get_indices()
        else:
            self.row_offs, self.col_offs = self._get_indices()
            self.num_chips = len(self.row_offs)

    def _get_bounds(self):
        if self.aoi is None:
            return self.raster_bounds

        return gpd.overlay(
            self._get_gpdf_from_bounds(self.raster_bounds),
            self._get_gpdf_from_bounds(self.aoi.total_bounds)
        ).total_bounds

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
                buf = self.feature_chip_size

            crs = self.profile['crs']
            buffed_gds = aoi.to_crs(crs).buffer(buf)
            return gpd.GeoDataFrame(geometry=buffed_gds)

        return None

    def _points_in_aoi(self, points: GeoSeries) -> GeoSeries:
        if self.aoi is None:
            good_points = points.within(self.aoi.unary_union)
            points = points[good_points]
        return points

    def _get_grid_points(self):
        x_min, y_min, x_max, y_max = self.bounds
        points = GeoSeries([
            Point(x, y)
            for x in range_with_end(
                x_min,
                x_max - self.feature_chip_size,
                self.output_chip_size
            )
            for y in range_with_end(
                y_min,
                y_max - self.feature_chip_size,
                self.output_chip_size
            )
        ])
        return self._points_in_aoi(points)

    def _get_random_points(self):
        x_min, y_min, x_max, y_max = self.bounds

        upper_left_points = []
        while len(upper_left_points) < self.num_chips:
            x = np.random.uniform(x_min, x_max, self.num_chips)
            y = np.random.uniform(y_min, y_max, self.num_chips)
            points = gpd.points_from_xy((x, y))
            upper_left_points.append(self.points_in_aoi(points))
        return upper_left_points[0:self.num_chips]

    def _get_indices(self):
        if self.mode == "train":
            upper_left_points = self._get_grid_points()
        else:
            upper_left_points = self._get_random_points()

        transform = self.profile['transform']
        return rowcol(transform, upper_left_points.x, upper_left_points.y)

    def _get_window(self, idx: int) -> Window:
        return Window(
            self.col_offs[idx],
            self.row_offs[idx],
            self.feature_chip_size,
            self.feature_chip_size
        )

    def _center_crop_window(self, window: Window, size: int) -> Window:
        col_off = window.col_off + window.width // 2 - (size // 2)
        row_off = window.row_off + window.height // 2 - (size // 2)
        return Window(col_off, row_off, size, size)

    def get_cropped_window(self, idx: int, size: int) -> Window:
        return self._center_crop_window(self._get_window(idx), size)

    def _get_gpdf_from_bounds(self, bounds: tuple) -> GeoDataFrame:
        return GeoDataFrame(
            {'geometry': [box(*bounds)]},
            crs=self.profile['crs']
        )

    def _get_gpdf_from_window(self, window: Window) -> GeoDataFrame:
        return self._get_gpdf_from_bounds(
            bounds(window, self.profile['transform']),
        )

    def _window_overlaps_aoi(self, window: Window) -> bool:
        if self.aoi is not None:
            window_gpdf = self._get_gpdf_from_window(window)
            overlap = gpd.overlay(window_gpdf, self.aoi)
            return overlap.shape[0] > 0

        return True

    @retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
    def _read_chip(self, ds, kwargs):
        return ds.read(**kwargs)

    def _get_img_chip(self, window: Window):
        with rio.open(self.feature_file) as img_ds:
            img_chip = self._read_chip(
                img_ds,
                {'indexes': range(1, 8), 'window': window, 'masked': True}
            ).filled(0) * img_ds.scales[0]
            # A couple leaps here ^^^^^
        return img_chip

    def _get_label_chip(self, window: Window):
        label_window = self._center_crop_window(
            window,
            self.output_chip_size
        )

        with rio.open(self.label_file) as label_ds:
            label_chip_raw = self._read_chip(
                label_ds,
                {'indexes': self.label_band, 'window': label_window}
            )

        imp_nodata_val = 127
        return masked_array(
            label_chip_raw,
            mask=label_chip_raw == imp_nodata_val
        ).filled(0) / 100
        # Replace nodatas with 0,
        # then divide by 100 for real values

    def __getitem__(self, idx: int):
        window = self._get_window(idx)

        img_chip = self._get_img_chip(window)

        if self.mode == 'train':
            label_chip = self._get_label_chip(window)
            return (img_chip, label_chip)

        return img_chip

    def __len__(self) -> int:
        return self.num_chips
