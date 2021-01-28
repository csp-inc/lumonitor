import os

from torch.utils.data import DataLoader

from StreamingZarrDataset import StreamingDataset

zarr_dir = 'hls_tmp'
# files = [ os.path.join(zarr_dir, f) for f in os.listdir(zarr_dir) ]
image_files = [os.path.join(zarr_dir, '0.zarr')]
label_files = [os.path.join(zarr_dir, 'hm_11ULP.tif')]


szd = StreamingDataset(image_files, label_files, label_band='impervious', num_chips_per_tile=20)

loader = DataLoader(szd)

for i, data in enumerate(loader):
    print(i, data)
