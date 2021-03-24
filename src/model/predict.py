import math
import os
import re

import numpy as np
import rioxarray
import torch
import torchvision.transforms.functional as tvF
import xarray as xr

from Unet_padded import Unet

cog_dir = 'data/cog/2016/training'
image_files = [
    os.path.join(cog_dir, f)
    for f in os.listdir(cog_dir)
    if not f.startswith('hm')
]

image_files = [
    os.path.join(cog_dir, '11SLT.tif'),
    os.path.join(cog_dir, '11SMT.tif'),
    os.path.join(cog_dir, '10SGJ.tif'),
]

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

N_BANDS = 7

model_root = 'imp_model_padded_2'
model_file = f'data/{model_root}.pt'
model = Unet(N_BANDS)

CHIP_SIZE = 512
test_chip = torch.Tensor(1, N_BANDS, CHIP_SIZE, CHIP_SIZE)
total_padding = CHIP_SIZE - 70
uncropped_padding = 100
OUTPUT_CHIP_SIZE = model.forward(test_chip).shape[2] - total_padding
padding = math.ceil((CHIP_SIZE - OUTPUT_CHIP_SIZE) / 2)

model.load_state_dict(torch.load(model_file))
net = model.float().to(dev)
model.eval()

def get_one_band_copy(ds):
    ods = ds.copy(deep=True).sel(band=[1])
    ods.attrs['scales'] = tuple([ods.attrs['scales'][0]])
    ods.attrs['nodatavals'] = tuple([ods.attrs['nodatavals'][0]])
    ods.attrs['offsets'] = tuple([ods.attrs['offsets'][0]])
    return ods

def get_output_file(input_file):
    output_file = re.sub('.tif', f'_pred_{model_root}.tif', input_file)
    output_dir = 'data/cog/2016/prediction'
    return os.path.join(output_dir, os.path.basename(output_file))


for image_file in image_files:
    img_ds = xr.open_rasterio(image_file, cache=True).fillna(0)
    output_ds = get_one_band_copy(img_ds)
    uncropped_output_ds = get_one_band_copy(img_ds)
    img_ds = torch.tensor(img_ds.values)

    n_rows = img_ds.shape[1]
    n_cols = img_ds.shape[2]

    n_chip_rows = math.ceil(n_rows / OUTPUT_CHIP_SIZE)
    n_chip_cols = math.ceil(n_cols / OUTPUT_CHIP_SIZE)

    for row in range(n_chip_rows):
        for col in range(n_chip_cols):
            xmin = 0 + col * OUTPUT_CHIP_SIZE
            xmax = xmin + CHIP_SIZE
            if xmax > n_cols:
                xmax = n_cols
                xmin = xmax - CHIP_SIZE

            ymin = 0 + row * OUTPUT_CHIP_SIZE
            ymax = ymin + CHIP_SIZE
            if ymax > n_rows:
                ymax = n_rows
                ymin = ymax - CHIP_SIZE

            input_data = img_ds[:, xmin:xmax, ymin:ymax].unsqueeze(0)
            output_data = model(input_data.float().to(dev))
            output_array = output_data.detach().cpu().numpy()
            ncol = np.shape(output_array)[1]
            nrow = np.shape(output_array)[2]

            if row == 0 or row == n_chip_rows - 1 or col == 0 or col == n_chip_cols - 1:
                uncropped_output_ds[
                    :,
                    (xmin + uncropped_padding):(xmax - uncropped_padding),
                    (ymin + uncropped_padding):(ymax - uncropped_padding)
                    ] = output_array[
                            :,
                            uncropped_padding:(ncol - uncropped_padding),
                            uncropped_padding:(nrow - uncropped_padding)
                        ]

            cropped_output_array = output_array[
                :, padding:ncol-padding, padding:nrow-padding
            ]
            output_ds[
                :, xmin+padding:xmax-padding, ymin+padding:ymax-padding
            ] = cropped_output_array

    uncropped_output_ds[
        :,
        padding:n_cols-padding,
        padding:n_rows-padding
    ] = output_ds[
        :,
        padding:n_cols-padding,
        padding:n_rows-padding
    ]

    uncropped_output_ds[:, 0:uncropped_padding, :] = -9999
    uncropped_output_ds[:, :, 0:uncropped_padding] = -9999
    uncropped_output_ds[:, (n_cols - uncropped_padding):n_cols, :] = -9999
    uncropped_output_ds[:, :, (n_rows - uncropped_padding):n_rows] = -9999

    output_file = get_output_file(image_file)
    uncropped_output_ds.rio.to_raster(output_file)
    print(output_file)
