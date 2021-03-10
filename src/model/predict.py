import math
import os
import re

import xarray as xr
import rioxarray
import torch
import torchvision.transforms.functional as tvF

from Unet import Unet

cog_dir = 'data/cog/2016/training'
image_files = [
    os.path.join(cog_dir, f)
    for f in os.listdir(cog_dir)
    if not f.startswith('hm')
]

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

model_file = 'data/hall_model6.pt'
model = Unet(7)
model.load_state_dict(torch.load(model_file))
net = model.float().to(dev)
model.eval()

chip_size = 512
output_size = 70
padding = math.ceil((chip_size - output_size) / 2)

def get_one_band_copy(ds):
    ods = ds.copy(deep=True).sel(band=[1])
    ods.attrs['scales'] = tuple([ods.attrs['scales'][0]])
    ods.attrs['nodatavals'] = tuple([ods.attrs['nodatavals'][0]])
    ods.attrs['offsets'] = tuple([ods.attrs['offsets'][0]])
    return ods

def get_output_file(input_file):
    output_file = re.sub('.tif', '_pred.tif', input_file)
    output_dir = 'data/cog/2016/prediction'
    return os.path.join(output_dir, os.path.basename(output_file))

for image_file in image_files:
    img_ds = xr.open_rasterio(image_file).fillna(0)
    output_ds = get_one_band_copy(img_ds)
    img_ds = torch.tensor(img_ds.values)

    n_rows = img_ds.shape[1]
    n_cols = img_ds.shape[2]
    n_chip_rows = math.ceil(n_rows / output_size)
    n_chip_cols = math.ceil(n_cols / output_size)

    for row in range(n_chip_rows):
        for col in range(n_chip_cols):
            xmin = 0 + row * output_size
            xmax = xmin + chip_size
            if xmax > n_rows:
                xmax = n_rows
                xmin = xmax - chip_size

            ymin = 0 + col * output_size
            ymax = ymin + chip_size
            if ymax > n_cols:
                ymax = n_cols
                ymin = ymax - chip_size

            input_data = img_ds[:, xmin:xmax, ymin:ymax].unsqueeze(0)
#            input_data = tvF.pad(input_data, math.ceil(buffer / 2))
            output_data = model(input_data.float().to(dev))
            output_ds[:, xmin+padding:xmax-padding, ymin+padding:ymax-padding] = output_data.detach().cpu().numpy()

    output_file = get_output_file(image_file)
    print(output_file)
    output_ds.rio.to_raster(output_file)

