import os

from azureml.core import Run
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
import torch
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet


import numpy as np

run = Run.get_context()
offline = run._run_id.startswith("OfflineRun")
path = 'data/azml' if offline else 'model/data/azml'

feature_file = os.path.join(path, 'conus_hls_median_2016.vrt')
output_file = 'outputs/predict_test.tif'

aoi = gpd.read_file(os.path.join(path, 'swatches.gpkg'))
aoi = aoi.iloc[[0]]

CHIP_SIZE = 512
OUTPUT_CHIP_SIZE = 70

pds = Dataset(
    feature_file,
    aoi=aoi,
    mode="predict"
)

print("num chips", pds.num_chips)
loader = DataLoader(pds)

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

model = Unet(7)
model_file = os.path.join(path, 'imp_model_padded_2.pt')
model.load_state_dict(torch.load(model_file, map_location=torch.device(dev)))
model.float().to(dev)
model.eval()

with rio.open(feature_file) as src:
    kwargs = src.meta.copy()
    out_ndarray, transform = mask(src, [box(*pds.bounds)], crop=True)


kwargs.update({
    'count': 1,
    'dtype': 'float32',
    'driver': 'GTiff',
    'bounds': pds.bounds,
    'height': out_ndarray.shape[1],
    'width': out_ndarray.shape[2],
    'transform': transform
})

with rio.open(output_file, 'w', **kwargs) as dst:
    for i, data in enumerate(loader):
        print(i)
        output_torch = model(data.float().to(dev))
        output_np = output_torch.detach().cpu().numpy()
        prediction = output_np[0:1, 221:291, 221:291]
        window = pds.get_cropped_window(i, OUTPUT_CHIP_SIZE, pds.aoi_transform)
        #pds._get_gpdf_from_window(window, pds.aoi_transform).to_file(f'window_{i}.shp')
        dst.write(prediction, window=window)
