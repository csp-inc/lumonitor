import argparse
import math
import os

from azureml.core import Run
import geopandas as gpd
from numpy import round
import rasterio as rio
import horovod.torch as hvd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet
from utils import get_device, get_output_specs
from utils.chippers import hm_chipper


def predict(aoi_file: str, feature_file: str, model_file: str) -> None:
    run = Run.get_context()
    offline = run._run_id.startswith("OfflineRun")
    path = "data/azml" if offline else "model/data/azml"

    aoi = gpd.read_file(os.path.join(path, aoi_file))

    OUTPUT_CHIP_SIZE = 70

    output_file = f"outputs/prediction_{hvd.rank()}.tif"

    feature_path = os.path.join(path, feature_file)
    pds = Dataset(feature_path, aoi=aoi, mode="predict")

    print("num chips", pds.num_chips)
    sampler = DistributedSampler(
        pds, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
    )
    loader = DataLoader(pds, batch_size=10, num_workers=6, sampler=sampler)

    dev = get_device()

    with rio.open(feature_path) as src:
        num_bands = src.count

    model = Unet(num_bands)

    model_path = os.path.join(path, model_file)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(dev)))
    model.half().to(dev)
    model.eval()
    bounds, height, width, transform = get_output_specs(feature_path, pds)

    with rio.open(feature_path) as src:
        kwargs = src.meta.copy()

    kwargs.update(
        {
            "blockxsize": 256,
            "blockysize": 256,
            "bounds": bounds,
            "compress": "LZW",
            "count": 1,
            "dtype": "uint8",
            "driver": "GTiff",
            "height": height,
            "nodata": 127,
            "predictor": 2,
            "tiled": "YES",
            "transform": transform,
            "width": width,
        }
    )

    with rio.open(output_file, "w", **kwargs) as dst:
        for _, (ds_idxes, data) in enumerate(loader):
            output = model(data.half().to(dev))
            data.detach()
            output_np = round(output.detach().cpu().numpy() * 100).astype("uint8")

            for j, idx_tensor in enumerate(ds_idxes):
                idx = idx_tensor.detach().cpu().numpy()
                if j % 1000 == 0:
                    print(idx)
                if len(output_np.shape) > 3:
                    prediction = output_np[j, 0:1, 221:291, 221:291]
                else:
                    prediction = output_np[0:1, 221:291, 221:291]

                window = pds.get_cropped_window(
                    idx, OUTPUT_CHIP_SIZE, pds.aoi_transform
                )
                dst.write(prediction, window=window)


if __name__ == "__main__":
    hvd.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi-file")
    parser.add_argument("--feature-file", help="Stem of the feature file")
    parser.add_argument("--model-file")
    args = parser.parse_args()

    predict(args.aoi_file, args.feature_file, args.model_file)
