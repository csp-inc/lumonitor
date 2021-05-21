import argparse
import math
import os

from affine import Affine
from azureml.core import Run, Workspace
import geopandas as gpd
from numpy import round
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import box
from tenacity import retry, stop_after_attempt, wait_fixed
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet


@retry(stop=stop_after_attempt(50), wait=wait_fixed(2))
def write(output_file, kwargs, prediction, window):
    with rio.open(
            output_file,
            'w',
            **kwargs,
            compress='LZW',
            predictor=2,
            blockxsize=256,
            blockysize=256
    ) as dst:
        dst.write(output_file, kwargs, prediction, window)


def get_output_specs(raster_file, dataset):
    """Used to do this with rasterio mask, but that required
       reading the whole file"""
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

    transform = Affine(
        transform.a,
        transform.b,
        xmin,
        transform.d,
        transform.e,
        ymax
    )
        
    return (xmin, ymin, xmax, ymax), height, width, transform

def predict(model_id: str) -> None:
    run = Run.get_context()
    offline = run._run_id.startswith("OfflineRun")
    path = 'data/azml' if offline else 'model/data/azml'

    feature_file = os.path.join(path, 'conus_hls_median_2016.vrt')
    model_id = 'lumonitor-conus-impervious-2016_1620952711_8aebb74b'

#    state_file = os.path.join(path, 'cb_2019_us_state_5m.zip')
#    states = gpd.read_file(state_file)
#    aoi = states[states['NAME'] == 'Vermont']
    conus_file = os.path.join(path, 'conus.geojson')
    aoi = gpd.read_file(conus_file)

    pds = Dataset(
        feature_file,
        aoi=aoi,
        mode="predict"
    )

    OUTPUT_CHIP_SIZE = 70

    if "WORLD_SIZE" in os.environ.keys():
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if world_size > 1:
        rank = int(os.environ["RANK"])
        output_file = f'outputs/prediction_{rank}.tif'
        chips_per_process = math.ceil(pds.num_chips / world_size)
        chip_start = rank * chips_per_process
        chip_end = min(pds.num_chips, chip_start + chips_per_process)
    else:
        output_file = 'outputs/prediction.tif'
        chip_start = 0
        chip_end = pds.num_chips

    pds.subset(chip_start, chip_end)

    print("total num chips", pds.num_chips)
    print("chip start ", chip_start)
    print("chip end ", chip_end)
    print("chips for rank ", chip_end - chip_start)
    loader = DataLoader(pds, batch_size=10, num_workers=6)

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    model = Unet(7)
    model_file = os.path.join(path, f'{model_id}.pt')

    model.load_state_dict(torch.load(model_file, map_location=torch.device(dev)))
    model.half().to(dev)
    model.eval()

    with rio.open(feature_file) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'count': 1,
            'dtype': 'uint8',
            'driver': 'GTiff',
            'nodata': 0
        })

    bounds, height, width, transform = get_output_specs(feature_file, pds)

    kwargs.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'bounds': bounds,
        'height': height,
        'width': width,
        'transform': transform,
        'nodata': 127
    })

    for _, (ds_idxes, data) in enumerate(loader):
        output = model(data.half().to(dev))
        data.detach()
        output_np = round(output.detach().cpu().numpy() * 100).astype('uint8')
        for j, idx_tensor in enumerate(ds_idxes):
            idx = idx_tensor.detach().cpu().numpy()
            if idx % 1000 == 0:
                print(idx)
            if len(output_np.shape) > 3:
                prediction = output_np[j, 0:1, 221:291, 221:291]
            else:
                prediction = output_np[0:1, 221:291, 221:291]

            window = pds.get_cropped_window(
                idx,
                OUTPUT_CHIP_SIZE,
                pds.aoi_transform
            )
            write(output_file, kwargs, prediction, window)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help="Stem of the model file")
    args = parser.parse_args()
    predict(args.model_id)
