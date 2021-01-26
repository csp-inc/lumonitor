import os

from torch.utils.data import DataLoader

from StreamingZarrDataset import StreamingZarrDataset


zarr_dir = 'hls_tmp'
files = [ os.path.join(zarr_dir, f) for f in os.listdir(zarr_dir) ]

szd = StreamingZarrDataset(files, num_chips_per_tile=20)

loader = DataLoader(szd)

for i, data in enumerate(loader):
    print(i, data)
