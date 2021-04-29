import geopandas as gpd
import rasterio as rio
import torch
from torch.utils.data import DataLoader

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet

feature_file = 'data/azml/conus_hls_median_2016.vrt'
#feature_file = 'data/hls_clip_test4.tif'
#aoi_file = 'data/azml/conus.geojson'
output_file = 'test_vermont.tif'

aoi = gpd.read_file('zip+http://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_state_5m.zip')
aoi = aoi[aoi.NAME == 'Vermont']
#aoi = gpd.read_file(aoi_file)

CHIP_SIZE = 512
OUTPUT_CHIP_SIZE = 70

pds = Dataset(
    feature_file,
    aoi=aoi,
    mode="predict"
)

loader = DataLoader(pds)

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

model = Unet(7)
model_file = 'data/imp_model_padded_2.pt'
model.load_state_dict(torch.load(model_file, map_location=torch.device(dev)))
model.float().to(dev)
model.eval()


with rio.open(feature_file) as src:
    kwargs = src.meta.copy()

kwargs.update({'count': 1, 'dtype': 'float32'})

with rio.open(output_file, 'w', **kwargs) as dst:
    for i, data in enumerate(loader):
        print(i)
        output_torch = model(data.float().to(dev))
        output_np = output_torch.detach().cpu().numpy()
        prediction = output_np[0:1, 221:291, 221:291]

        window = pds.get_cropped_window(i, OUTPUT_CHIP_SIZE)
        dst.write(prediction, window=window)
