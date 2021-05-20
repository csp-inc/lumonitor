import argparse
import math
import os

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

    OUTPUT_CHIP_SIZE = 70

    world_size = int(os.environ["WORLD_SIZE"])

    output_file = 'outputs/prediction.tif'

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
        aoi.to_file(f'outputs/prediction_{rank}.shp')
        output_file = f'outputs/prediction_{rank}.tif'

    pds = Dataset(
        feature_file,
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

#    with MemoryFile() as memfile:
#        with memfile.open(**kwargs) as src:
            # Causes a read of src in the entirety of the box
            # using memfile works, but for conus it may be too big (is there a "bit" datatype?)
#            out_ndarray, transform = mask(src, [box(*pds.bounds)], crop=True)

    kwargs.update({
        'count': 1,
        'dtype': 'uint8',
        'driver': 'GTiff',
#        'bounds': pds.bounds,
#        'height': out_ndarray.shape[1],
#        'width': out_ndarray.shape[2],
#        'transform': transform,
        'nodata': 127
    })

#    with rio.open(output_file, 'w', **kwargs) as dst:
#        for _, (ds_idxes, data) in enumerate(loader):
#            output = model(data.half().to(dev))
#            data.detach()
#            output_np = round(output.detach().cpu().numpy() * 100).astype('uint8')
#
#            for j, idx_tensor in enumerate(ds_idxes):
#                idx = idx_tensor.detach().cpu().numpy()
#                if idx % 1000 == 0:
#                    print(idx)
#                if len(output_np.shape) > 3:
#                    prediction = output_np[j, 0:1, 221:291, 221:291]
#                else:
#                    prediction = output_np[0:1, 221:291, 221:291]
#
#                window = pds.get_cropped_window(
#                    idx,
#                    OUTPUT_CHIP_SIZE,
#                    pds.aoi_transform
#                )
#                dst.write(prediction, window=window)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help="Stem of the model file")
    args = parser.parse_args()
    predict(args.model_id)
