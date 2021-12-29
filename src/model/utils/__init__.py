import math
import os
from typing import Tuple

from affine import Affine
from azureml.core import Environment
import rasterio as rio
import torch

from MosaicDataset import MosaicDataset as Dataset


def get_device(use_hvd: bool = True) -> torch.device:
    """Returns the device"""
    if use_hvd:
        import horovod.torch as hvd

        hvd.init()

    # As long as there is only 1 gpu per node this will always be 0
    local_rank = hvd.local_rank() if use_hvd else 0

    return torch.device(
        "cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"
    )


def load_azml_env(img_tag: str = "latest") -> Environment:
    env = Environment("lumonitor")
    env.docker.base_image = f"cspincregistry.azurecr.io/lumonitor-azml:{img_tag}"
    env.python.user_managed_dependencies = True
    env.docker.base_image_registry.address = "cspincregistry.azurecr.io"
    env.docker.base_image_registry.username = os.environ["AZURE_REGISTRY_USERNAME"]
    env.docker.base_image_registry.password = os.environ["AZURE_REGISTRY_PASSWORD"]

    env.environment_variables = dict(
        AZURE_STORAGE_ACCOUNT=os.environ["AZURE_STORAGE_ACCOUNT"],
        AZURE_STORAGE_ACCESS_KEY=os.environ["AZURE_STORAGE_ACCESS_KEY"],
    )

    return env


def get_output_specs(raster_file: str, dataset: Dataset) -> Tuple:
    """For use in prediction"""
    # Used to do this with rasterio mask, but that required
    #   reading the whole file
    with rio.open(raster_file) as src:
        image_xmin, image_ymin, _, _ = src.bounds
        res = src.res[0]
        transform = src.transform

    area_xmin, area_ymin, area_xmax, area_ymax = dataset.bounds

    xmin = image_xmin + ((area_xmin - image_xmin) // res) * res
    xmax = xmin + math.ceil((area_xmax - xmin) / res) * res
    # this _should_ be a multiple of res
    width = int((xmax - xmin) / res)

    ymin = image_ymin + ((area_ymin - image_ymin) // res) * res
    ymax = ymin + math.ceil((area_ymax - ymin) / res) * res
    height = int((ymax - ymin) / res)

    transform = Affine(transform.a, transform.b, xmin, transform.d, transform.e, ymax)

    return (xmin, ymin, xmax, ymax), height, width, transform
