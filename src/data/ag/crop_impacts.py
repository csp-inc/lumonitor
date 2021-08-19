import argparse
from math import prod

import dask
import xarray as xr
import rioxarray as rx

from dask.distributed import Client, Lock


def binary_to_acres(binary: xr.DataArray) -> xr.DataArray:
    """
    Converts a 1/0 image to the acres of 1 in each cell. Cell units must be in
    square meters
    """
    sq_meters_per_cell = abs(prod(binary.rio.resolution))
    acres_per_sq_meter = 2.47105e-6
    acres_per_cell = sq_meters_per_cell * acres_per_sq_meter
    return binary * acres_per_cell

def acre_feet_per_cell(acre_feet_per_acres: xr.DataArray, irrigants: xr.DataArray) -> xr.DataArray:
    total_acres = irrigants.
