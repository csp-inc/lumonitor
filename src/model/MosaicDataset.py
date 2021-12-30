from math import ceil, floor, sqrt
from typing import Callable, Generator, Tuple, Union

from affine import Affine
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
from numpy.ma import masked_array
from pandas import Series
from pygeos.creation import points, prepare
import pygeos
import rasterio as rio
from rasterio.windows import bounds, Window
from rasterio.transform import rowcol
from shapely.geometry import box, Point
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from torch.utils.data.dataset import Dataset


def range_with_end(start: int, end: int, step: int) -> Generator[int, None, None]:
    """Returns a generator for the given parameters, which also includes the end
    value"""
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
        output_chip_size: int = 512,
        unpadded_chip_size: int = 70,
        aoi: GeoDataFrame = None,
        buffer_aoi: bool = True,
        label_file: str = None,
        label_bands: int = 1,
        label_chip_from_raw_chip: Callable = None,
        mode: str = "train",  # Either "train" or "predict"
        num_training_chips: int = 0,
    ):
        self.feature_file = feature_file
        self.label_chip_from_raw_chip = label_chip_from_raw_chip
        self.feature_chip_size = feature_chip_size
        self.output_chip_size = output_chip_size
        self.unpadded_chip_size = unpadded_chip_size
        self.mode = mode
        (
            self.raster_bounds,
            self.input_transform,
            self.crs,
            self.res,
        ) = self._get_raster_info()
        self.buffer_aoi = buffer_aoi
        self.aoi = self._transform_and_buffer(aoi)
        self.bounds, self.aoi_transform = self._get_bounds()

        if self.mode == "train":
            self.label_file = label_file
            self.label_bands = label_bands
            # These need to happen in these orders :eyeroll:
            self.num_chips = num_training_chips
            self.chip_xs, self.chip_ys = self._get_indices()
        else:
            self.chip_xs, self.chip_ys = self._get_indices()
            self.num_chips = len(self.chip_xs)

    def _get_bounds(self) -> tuple:
        if self.aoi is None:
            return self.raster_bounds, self.input_transform

        overlay = gpd.overlay(
            self._get_gpdf_from_bounds(self.raster_bounds),
            self._get_gpdf_from_bounds(self.aoi.total_bounds),
        )

        xmin, _, _, ymax = overlay.total_bounds

        aoi_transform = Affine(
            self.input_transform.a,
            self.input_transform.b,
            xmin,
            self.input_transform.d,
            self.input_transform.e,
            ymax,
        )

        return overlay.total_bounds, aoi_transform

    def _get_raster_info(self):
        with rio.open(self.feature_file) as r:
            return r.bounds, r.transform, r.crs, r.res[0]

    def _transform_and_buffer(self, aoi: GeoDataFrame) -> GeoDataFrame:
        """Buffer the aoi. If training, then we take a negative buffer, equal
        to the smallest whole number closest to one half the chip size, so chips
        always fall completely in the aoi. Otherwise, it's a positive buffer
        equal to the distance from the center of a chip to any corner, ensuring
        that the full aoi is covered by all chips in a grid"""
        if aoi is not None:
            aoi = aoi.to_crs(self.crs)

            if self.buffer_aoi:
                if self.mode == "train":
                    buf = floor(-1 * (self.feature_chip_size / 2) * self.res)
                else:
                    buf = (self.feature_chip_size / 2) * self.res * sqrt(2)

                buffed_gds = aoi.buffer(buf)
                return gpd.GeoDataFrame(geometry=buffed_gds)

            return aoi

        return None

    def _get_indices(self) -> Tuple[Series, Series]:
        """Returns a tuple of Geoseries objects with the upper left coordinates
        of all chips"""
        upper_left_points = (
            self._get_random_points()
            if self.mode == "train"
            else self._get_grid_points()
        )

        return upper_left_points.x, upper_left_points.y

    def _get_random_points(self, n: int = None) -> GeoSeries:
        """Returns a Geoseries with the upper left corner of n points whose
        entire extents are within the AOI"""
        if n is None:
            n = self.num_chips

        x_min, y_min, x_max, y_max = self.bounds

        upper_left_points = GeoSeries([])
        while len(upper_left_points) < self.num_chips:
            n_to_pull = max(n * 2, 10000)
            x = np.random.uniform(x_min, x_max, n_to_pull)
            y = np.random.uniform(y_min, y_max, n_to_pull)
            new_points = self._points_in_aoi(points(x, y))
            upper_left_points = upper_left_points.append(new_points, ignore_index=True)
            # print(f"{len(upper_left_points)} / {self.num_chips} points")

        return upper_left_points[0:n]

    def _get_grid_points(self) -> GeoSeries:
        """Returns a Geoseries of points of upper left corners to cover the entire
        bounds"""
        x_min, y_min, x_max, y_max = self.bounds

        feature_chip_size_map = self.feature_chip_size * self.res
        unpadded_chip_size_map = self.unpadded_chip_size * self.res

        pts = [
            (x, y)
            for x in range_with_end(
                x_min,
                x_max - ceil(feature_chip_size_map),
                floor(unpadded_chip_size_map),
            )
            for y in range_with_end(
                y_min + ceil(feature_chip_size_map),
                y_max,
                floor(unpadded_chip_size_map),
            )
        ]

        return self._points_in_aoi(points(pts))

    def _replace_index(self, idx: int) -> None:
        """Replace the point at idx with a new one. Supposed to be used to
        'repair' errors for certain points, but in actuality the errors are
        often due to network I/O cutouts."""
        print(f"replacing point at index {idx}")
        point = self._get_random_points(1)
        self.chip_xs[idx], self.chip_ys[idx] = point.x[0], point.y[0]

    def _points_in_aoi(self, pts: np.ndarray) -> GeoSeries:
        """If self.aoi is set, returns the subset of points which are within
        it. Otherwise just returns pts. Note pts is typically created by
        pygeos.creation.points"""
        if self.aoi is not None:
            aoi = pygeos.io.from_shapely(self.aoi.unary_union)
            prepare(aoi)
            prepare(pts)
            good_pts = pygeos.contains(aoi, pts)
            pts = pts[good_pts]
        return GeoSeries(pts)

    def _get_window(self, idx: int, transform: Affine = None) -> Window:
        """Returns a Window in the given transformation with sides of
        self.feature_chip_size and with an upper left corner self.chip_xs[idx],
        self.chip_ys[idx]"""
        if transform is None:
            transform = self.input_transform

        row_off, col_off = rowcol(transform, self.chip_xs[idx], self.chip_ys[idx])

        return Window(col_off, row_off, self.feature_chip_size, self.feature_chip_size)

    def _center_crop_window(self, window: Window, size: int) -> Window:
        """For a given window, crops to the central size x size square"""
        col_off = window.col_off + window.width // 2 - (size // 2)
        row_off = window.row_off + window.height // 2 - (size // 2)
        return Window(col_off, row_off, size, size)

    def get_cropped_window(
        self, idx: int, size: int, transform: Affine = None
    ) -> Window:
        """Crops the center of the window at the given index to a square
        with sides of size"""
        return self._center_crop_window(self._get_window(idx, transform), size)

    def _get_gpdf_from_bounds(self, bounds: tuple) -> GeoDataFrame:
        """Returns a GeoDataFrame of the given bounds"""
        return GeoDataFrame({"geometry": [box(*bounds)]}, crs=self.crs)

    def _get_gpdf_from_window(self, window: Window, transform) -> GeoDataFrame:
        """Returns a GeoDataFrame of the bounds of the given window"""
        return self._get_gpdf_from_bounds(bounds(window, transform))

    #    @retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
    def _read_chip(self, ds: rio.DatasetReader, kwargs) -> np.ndarray:
        """Reads from the given dataset with the parameters in kwargs. Created
        mostly to utilize retry functionality"""
        return ds.read(**kwargs)

    #    @retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
    def _get_img_chip(self, window: Window) -> np.ndarray:
        """Returns the feature data for the given window"""
        with rio.open(self.feature_file) as img_ds:
            img_chip = (
                self._read_chip(img_ds, {"window": window, "masked": True}).filled(0)
                * img_ds.scales[0]
            )
            # A couple leaps here ^^^^^
        return np.nan_to_num(img_chip, copy=False)

    def _get_label_chip(self, window: Window):
        """Returns the label data for the given window"""
        label_window = self._center_crop_window(window, self.output_chip_size)

        with rio.open(self.label_file) as label_ds:
            label_chip_raw = self._read_chip(
                label_ds, {"indexes": self.label_bands, "window": label_window}
            )

        return np.nan_to_num(self.label_chip_from_raw_chip(label_chip_raw), copy=False)

    def _get_chip_for_idx(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Returns either the feature data or the feature and label data for
        the given index"""
        window = self._get_window(idx)
        img_chip = self._get_img_chip(window)

        if self.mode == "train":
            label_chip = self._get_label_chip(window)
            return (img_chip, label_chip)

        return img_chip

    def __getitem__(self, idx: int):
        try:
            item = self._get_chip_for_idx(idx)
        except RetryError as e:
            if self.mode == "train":
                self._replace_index(idx)
                item = self._get_chip_for_idx(idx)
            else:
                raise e

        return (idx, item)

    def __len__(self) -> int:
        return self.num_chips

    def subset(self, start_idx: int, end_idx: int) -> None:
        """Subsets the Dataset based on the given indices"""
        self.chip_xs = self.chip_xs[start_idx:end_idx].reset_index(drop=True)
        self.chip_ys = self.chip_ys[start_idx:end_idx].reset_index(drop=True)
        self.num_chips = len(self.chip_xs)
