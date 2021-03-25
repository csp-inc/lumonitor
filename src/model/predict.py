import math
import os
import re

import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import torch
import torchvision.transforms.functional as tvF

from Unet_padded import Unet

def get_output_file(input_file):
    output_file = re.sub('.tif', f'_pred_{model_root}.tif', input_file)
    output_dir = 'data/cog/2016/prediction'
    return os.path.join(output_dir, os.path.basename(output_file))

def write_output_file(
        input_rio,
        output_np,
        output_file,
        dst_crs='EPSG:4326'
        ):

    # EVERYWHERE else it's height, width
    transform, width, height = calculate_default_transform(
        src_crs=input_rio.crs,
        dst_crs=dst_crs,
        width=output_np.shape[1],
        height=output_np.shape[2],
        left=input_rio.bounds.left,
        bottom=input_rio.bounds.bottom,
        right=input_rio.bounds.right,
        top=input_rio.bounds.top,
        dst_width=output_np.shape[1],
        dst_height=output_np.shape[2]
    )

    kwargs = input_rio.meta.copy()

    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    output_rio = rio.open(
        output_file,
        mode='w',
        driver='GTiff',
        width=width,
        height=height,
        count=output_np.shape[0],
        crs=dst_crs,
        transform=transform,
        # Possible we can change this to save space
        dtype=output_np.dtype,
        nodata=np.NaN,
        # kwargs
    )

    output_np_transformed = np.nan_to_num(np.zeros_like(output_np), copy=False)

    reproject(
        source=output_np,
        destination=output_np_transformed,
        src_transform=input_rio.transform,
        src_crs=input_rio.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        dst_nodata=np.NaN,
        resampling=Resampling.nearest
    )

    output_rio.write(output_np_transformed)
    output_rio.close()

cog_dir = 'data/cog/2016/training'
image_files = [
    os.path.join(cog_dir, f)
    for f in os.listdir(cog_dir)
    if not f.startswith('hm')
]

#image_files = [
#    os.path.join(cog_dir, '11SLT.tif'),
#    os.path.join(cog_dir, '11SMT.tif'),
#    os.path.join(cog_dir, '10SGJ.tif'),
#]

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


for image_file in image_files:
    input_rio = rio.open(image_file)
    input_np = np.nan_to_num(input_rio.read(), copy=False)
    n_rows = input_np.shape[1]
    n_cols = input_np.shape[2]

    output_np = np.zeros((1, n_rows, n_cols))
    output_np[:] = np.NaN

    uncropped_output_np = output_np.copy()

    input_t = torch.tensor(input_np)

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

            input_data = input_t[:, xmin:xmax, ymin:ymax].unsqueeze(0)
            output_data = model(input_data.float().to(dev))
            output_array = output_data.detach().cpu().numpy()

            if row == 0 or row == n_chip_rows - 1 or col == 0 or col == n_chip_cols - 1:
                uncropped_output_np[
                    :,
                    (xmin + uncropped_padding):(xmax - uncropped_padding),
                    (ymin + uncropped_padding):(ymax - uncropped_padding)
                    ] = output_array[
                        :,
                        uncropped_padding:(CHIP_SIZE - uncropped_padding),
                        uncropped_padding:(CHIP_SIZE - uncropped_padding)
                    ]

            cropped_output_array = output_array[
                :, padding:CHIP_SIZE-padding, padding:CHIP_SIZE-padding
            ]
            output_np[
                :, xmin+padding:xmax-padding, ymin+padding:ymax-padding
            ] = cropped_output_array

    uncropped_output_np[
        :,
        padding:n_cols-padding,
        padding:n_rows-padding
    ] = output_np[
        :,
        padding:n_cols-padding,
        padding:n_rows-padding
    ]

    uncropped_output_np[:, 0:uncropped_padding, :] = np.NaN
    uncropped_output_np[:, :, 0:uncropped_padding] = np.NaN
    uncropped_output_np[:, (n_cols - uncropped_padding):n_cols, :] = np.NaN
    uncropped_output_np[:, :, (n_rows - uncropped_padding):n_rows] = np.NaN

    output_file = get_output_file(image_file)

    write_output_file(input_rio, uncropped_output_np, output_file)
    print(output_file)
