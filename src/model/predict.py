import math
import os

from azureml.core import Run
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import box
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet

run = Run.get_context()
offline = run._run_id.startswith("OfflineRun")
path = 'data/azml' if offline else 'model/data/azml'

feature_file = os.path.join(path, 'conus_hls_median_2016.vrt')
output_file = 'outputs/predict_test.tif'

state_file = os.path.join(path, 'cb_2019_us_state_5m.zip')
states = gpd.read_file(state_file)
aoi = states[states['NAME'] == 'Texas']

CHIP_SIZE = 512
OUTPUT_CHIP_SIZE = 70

world_size = int(os.environ["WORLD_SIZE"])

if world_size > 1:
    xmin, ymin, xmax, ymax = aoi.total_bounds
    width = xmax - xmin
    height = ymax - ymin
    side_length = math.sqrt(height * width / world_size)
    n_cols = math.ceil(width / side_length)
    rank = int(os.environ["RANK"])
    this_row = rank // n_cols
    this_col = rank % n_cols
    i_xmin = xmin + this_row * side_length
    i_ymin = ymin + this_col * side_length
    i_xmax = i_xmin + side_length
    i_ymax = i_ymin + side_length
    aoi = gpd.GeoDataFrame(
        geometry=[box(i_xmin, i_ymin, i_xmax, i_ymax)],
        crs=aoi.crs
    )

    output_file = f'outputs/predict_test_{rank}.tif'

pds = Dataset(
    feature_file,
    aoi=aoi,
    mode="predict"
)

print("num chips", pds.num_chips)
loader = DataLoader(pds, batch_size=5, num_workers=6)

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

model = Unet(7)
model_file = os.path.join(
    path,
    'lumonitor-conus-impervious-2016_1620952711_8aebb74b.pt'
)
model.load_state_dict(torch.load(model_file, map_location=torch.device(dev)))
if dev == 'cuda':
    model = DataParallel(model)
model.working_type().to(dev)
model.eval()

with rio.open(feature_file) as src:
    kwargs = src.meta.copy()
    kwargs.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'nodata': 0
    })

with MemoryFile() as memfile:
    with memfile.open(**kwargs) as src:
        # Causes a read of src in the entirety of the box
        # using memfile works, but for conus it may be too big (is there a "bit" datatype?)
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
    for i, (ds_idxes, data) in enumerate(loader):
        output = model(data.half().to(dev))
        data.detach()
        output_np = output.detach().cpu().numpy()
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
