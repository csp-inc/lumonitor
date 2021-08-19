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
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet

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

def predict(model_id: str, aoi_file: str, feature_file: str) -> None:
    run = Run.get_context()
    offline = run._run_id.startswith("OfflineRun")
    path = 'data/azml' if offline else 'model/data/azml'

    model_id = 'lumonitor-conus-impervious-2016_1620952711_8aebb74b'

    aoi = gpd.read_file(aoi_file)

    OUTPUT_CHIP_SIZE = 70

#    number = re.sub('^.*_(.*)\..*$', '\\1', j)
    file_id = os.path.splitext(os.path.basename(aoi_file))[0]
    output_file = f'outputs/prediction_{file_id}.tif'

    feature_path = os.path.join(path, feature_file)
    pds = Dataset(
        feature_path,
        aoi=aoi,
        mode="predict"
    )

    print("num chips", pds.num_chips)
    loader = DataLoader(pds, batch_size=10, num_workers=6)

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    model = Unet(7)
    model_path = os.path.join(path, f'{model_id}.pt')

    model.load_state_dict(torch.load(model_path, map_location=torch.device(dev)))
    model.half().to(dev)
    model.eval()

    with rio.open(feature_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'count': 1,
            'dtype': 'uint8',
            'driver': 'GTiff',
            'nodata': 0
        })

    bounds, height, width, transform = get_output_specs(feature_path, pds)
    kwargs.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'bounds': bounds,
        'height': height,
        'width': width,
        'transform': transform,
        'nodata': 127,
        'compress': 'LZW',
        'predictor': 2,
        'blockxsize': 256,
        'blockysize': 256,
        'tiled': 'YES'
    })

    with rio.open(output_file, 'w', **kwargs) as dst:
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
                dst.write(prediction, window=window)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help="Stem of the model file")
    parser.add_argument('--aoi_file')
    parser.add_argument('--feature_file', help="Stem of the feature file")
    args = parser.parse_args()
    predict(args.model_id, args.aoi_file, args.feature_file)
