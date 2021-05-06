import argparse
import math
import os
import random
import re
import yaml
from dataclasses import dataclass

from azureml.core import Run
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

@dataclass
class Trainer():
    epochs: int
    training_samples: int
    test_samples: int
    learning_rate: float = 0.01
    learning_schedule_gamma: float = 0.975
    momentum: float = 0.8
    chip_size: int = 512
    aoi_file: str = 'conus.geojson'
    batch_size: int = 8
    num_workers: int = 6
    seed: int = 1337

    def __post_init__(self):
        self.run = Run.get_context()
        offline = self.run._run_id.startswith("OfflineRun")
        path = 'data/azml' if offline else 'model/data/azml'

        self.training_file = os.path.join(path, 'conus_hls_median_2016.vrt')
        self.label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'
        self.aoi = self._load_aoi(path, self.aoi_file)
        self.num_training_bands, self.res = self._get_training_raster_specs()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.dev)
        self.net = Unet(self.num_training_bands).float().to(self.dev)
        self.output_chip_size = self._get_output_chip_size()

        swatch_file = os.path.join(path, 'swatches.gpkg')
        self.swatches = gpd.read_file(swatch_file)

        self.dataset_kwargs = {
            'feature_file': self.training_file,
            'feature_chip_size': self.chip_size,
            'output_chip_size': self.output_chip_size,
            'aoi': self.aoi,
            'label_file': self.label_file
        }
        self.dataloader_kwargs = {
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'pin_memory': True,
            'worker_init_fn': worker_init_fn
        }
        self.test_ds = Dataset(
            num_training_chips=self.test_samples,
            **self.dataset_kwargs
        )
        self.test_loader = DataLoader(
            self.test_ds,
            shuffle=False,
            **self.dataloader_kwargs
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_lambda
        )

    def _get_lr_lambda(self, epoch: int) -> float:
        return self.learning_schedule_gamma ** epoch

    def _load_aoi(self, path, aoi_file: str) -> gpd.GeoDataFrame:
        if aoi_file is not None:
            aoi_file = os.path.join(path, aoi_file)
            return gpd.read_file(aoi_file)
        return None

    def _get_training_raster_specs(self) -> int:
        with rio.open(self.training_file) as src:
            num_bands = src.count
            res = src.res[0]

        return num_bands, res

    def _get_output_chip_size(self) -> int:
        # Not to be confused with "good" area
        test_chip = torch.Tensor(
            1,
            self.num_training_bands,
            self.chip_size,
            self.chip_size
        ).to(self.dev)
        return self.net.forward(test_chip).shape[2]

    def _get_ds(self) -> Dataset:
        return Dataset(
            num_training_chips=self.training_samples,
            **self.dataset_kwargs
        )

    def _get_loader(self, ds: Dataset) -> DataLoader:
        return DataLoader(
            ds,
            shuffle=True,
            **self.dataloader_kwargs
        )

    def train(self):
        for epoch in range(self.epochs):
            np.random.seed(self.seed)

            # Reset this at each epoch so samples change
            ds = self._get_ds()
            loader = self._get_loader(ds)

            running_loss = self._train_step(loader)

            train_loss = running_loss / self.training_samples
            print('Epoch %d loss: %.4f' % (epoch + 1, train_loss))

            with torch.no_grad():
                test_running_loss = self._eval_step()
                self._write_swatches(epoch)

            test_loss = test_running_loss / self.test_samples
            print('Epoch %d test loss: %.4f' % (epoch + 1, test_loss))

            run.log("epoch", epoch+1)
            run.log("Training Loss", train_loss)
            run.log("Test Loss", test_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            print('LR: %.4f' % lr)
            run.log("lr", lr)
            self.scheduler.step()

            # Just for testing, we dont' need to save GB of models
            # ^^ Well then how do we go back?
            # Only save the last one?
            torch.save(self.net.state_dict(), f'./outputs/model.pt')

    def _train_step(self, loader: DataLoader) -> float:
        running_loss = 0.0
        self.net.train()
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(self.dev)
            labels = labels.to(self.dev)
            self.optimizer.zero_grad()

            outputs = self.net(inputs.float())
            loss = self.criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * self.batch_size
        inputs.detach()
        labels.detach()
        return running_loss

    def _eval_step(self) -> float:
        test_running_loss = 0.0
        self.net.eval()
        for i, test_data in enumerate(self.test_loader):
            inputs, labels = test_data
            inputs = inputs.to(self.dev)
            labels = labels.to(self.dev)
            outputs = self.net(inputs.float())
            loss = self.criterion(outputs.squeeze(1), labels.float())
            test_running_loss += loss.item() * self.batch_size
        inputs.detach()
        labels.detach()
        return test_running_loss

    def _write_swatches(self, epoch: int) -> None:
        ramp = [
            (0, '#010101'),
            (1/3., '#ff0101'),
            (2/3., '#ffbb01'),
            (1, '#ffff01')
        ]
        cm = LinearSegmentedColormap.from_list("urban", ramp)
        swatch_dataloader_kwargs = self.dataloader_kwargs.copy()
        swatch_dataloader_kwargs.update({'batch_size': 1})
        self.net.eval()
        for i, _ in self.swatches.iterrows():
            swatch = self.swatches.loc[[i]]
            swatch_kwargs = self.dataset_kwargs.copy()
            swatch_kwargs.update({'aoi': swatch, 'mode': 'predict'})
            ds = Dataset(**swatch_kwargs)
            dl = DataLoader(ds) #, **swatch_dataloader_kwargs)
            with rio.open(self.training_file) as src:
                kwargs = src.meta.copy()
                out_array, transform = mask(src, [box(*ds.bounds)], crop=True)
            kwargs.update({
                'count': 1,
                'dtype': 'float32',
                'driver': 'GTiff',
                'bounds': ds.bounds,
                'height': out_array.shape[1],
                'width': out_array.shape[2],
                'transform': transform
            })

            output_file = f'outputs/swatch_{i}_{epoch}.tif'
            swatch_np = np.empty((out_array.shape[1], out_array.shape[2]))
            with rio.open(output_file, 'w', **kwargs) as dst:
                for idx, data in enumerate(dl):
                    output_tensor = self.net(data.float().to(self.dev))
                    output_np = output_tensor.detach().cpu().numpy()
                    prediction = output_np[0:1, 221:291, 221:291]
                    window = ds.get_cropped_window(idx, 70, ds.aoi_transform)
                    dst.write(prediction, window=window)
                    swatch_np[
                        window.col_off:window.col_off + window.width,
                        window.row_off:window.row_off + window.height
                    ] = prediction
            output_png = f'outputs/swatch_{i}_{epoch}.png'
            rgb = Image.fromarray(np.uint8(cm(swatch_np)*255))
            rgb.save(output_png)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params-path',
        help="Path to params file, see src/model/configs"
    )
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = yaml.safe_load(f)

    run = Run.get_context()
    for k, v in params.items():
        run.tag(k, str(v))

    trainer = Trainer(**params)
    trainer.train()
