import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform

def copy_with_changes(src_file, dst_file, dst_crs, dst_bands, dst_dtype):
    input_rio = rio.open(src_file)
    kwargs = input_rio.meta.copy()

    transform, width, height = calculate_default_transform(
        src_crs=input_rio.crs,
        dst_crs=dst_crs,
        width=input_rio.shape[0],
        height=input_rio.shape[1],
        left=input_rio.bounds.left,
        bottom=input_rio.bounds.bottom,
        right=input_rio.bounds.right,
        top=input_rio.bounds.top,
        dst_width=input_rio.shape[0],
        dst_height=input_rio.shape[1]
    )
    
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(
            dst_file,
            mode='w',
            driver='GTiff',
            width=width,
            height=height,
            count=dst_bands,
            crs=dst_crs,
            transform=transform,
            dtype=dst_dtype,
            nodata=np.NaN,
            compress='LZW',
            predictor=2,
            tiled=True
    ) as dst:
        dst.write(np.empty(height, width), indexes=range(1, dst_bands + 1))
