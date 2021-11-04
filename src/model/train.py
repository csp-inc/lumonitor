import argparse
from dataclasses import dataclass
import os
import random
from typing import Callable
import yaml

from azureml.core import Run
import geopandas as gpd
import numpy as np
from numpy.ma import masked_array
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame
from PIL import Image
import rasterio as rio
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def set_seed(seed: int) -> None:
    """Basically set the seed in all possible ways"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nlcd_chipper(chip_raw: np.ndarray) -> np.ndarray:
    # Don't know where these -1000s are coming from, but
    # they are there on read
    masked = (chip_raw > 100) | (chip_raw == -1000)
    # Replace nodatas with 0,
    # then divide by 100 for real values
    return (
        masked_array(
            chip_raw,
            mask=masked,
        ).filled(0)
        / 100.0
    )


def hm_chipper(chip_raw: np.ndarray) -> np.ndarray:
    masked = chip_raw == -32768
    return masked_array(chip_raw, mask=masked).filled(0) / 10000.0


@dataclass
class Trainer:
    epochs: int  # number of epochs
    training_samples: int  # number of training samples
    test_samples: int  # number of test samples
    use_hvd: bool  # whether to use horovod multiprocessing or not
    batch_size: int = 8
    chip_size: int = 512  # value of the w & h for each sample

    # Function to convert raw data read from the label file into the range and type of
    # training data. e.g. divide nlcd by 100 and convert to float
    chipper: Callable = nlcd_chipper

    # Source of the label data, should have the same projection, extent, etc
    # as the training file
    label_file: str = "/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif"

    learning_rate: float = 0.01  # Initial learning rate
    learning_schedule_gamma: float = 0.975
    momentum: float = 0.8

    # Number of gpus to use; only relevant if use_hvd is True
    num_gpus: int = 1

    # Number of workers each dataloader uses to read data
    num_workers: int = 6
    seed: int = 1337  # Random seed
    test_aoi_file: str = "conus.geojson"  # AOI within which to pull test data

    # Source of the training data
    training_file: str = "/vsiaz/hls/cog/conus_hls_median_2016.vrt"

    # AOI within which to pull training data
    training_aoi_file: str = "conus.geojson"

    def __post_init__(self):
        self.run = Run.get_context()
        offline = self.run._run_id.startswith("OfflineRun")
        path = "data/azml" if offline else "model/data/azml"
        self.training_aoi = self._load_aoi(path, self.training_aoi_file)
        self.test_aoi = self._load_aoi(path, self.test_aoi_file)
        self.num_training_bands, self.res = self._get_raster_specs(self.training_file)
        self.num_label_bands, _ = self._get_raster_specs(self.label_file)

        if self.use_hvd:
            # Scale the learning rate based on the number of gpus,
            # see https://horovod.readthedocs.io/en/stable/pytorch.html
            self.learning_rate *= self.num_gpus

        # As long as there is only 1 gpu per node this will always be 0
        local_rank = hvd.local_rank() if self.use_hvd else 0

        self.dev = torch.device(
            "cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.dev)
        set_seed(self.seed)

        self.net = (
            Unet(self.num_training_bands, self.num_label_bands).float().to(self.dev)
        )
        self.output_chip_size = self._get_output_chip_size()

        self._loss = None
        self._test_loss = None
        self._lr = None
        self._epoch = None
        self.log_df = DataFrame(columns=["epoch", "loss", "test_loss", "lr"])

        swatch_file = os.path.join(path, "swatches.gpkg")
        self.swatches = gpd.read_file(swatch_file)
        label_bands = (
            range(1, self.num_label_bands + 1) if self.num_label_bands > 1 else 1
        )

        self.dataset_kwargs = {
            "feature_file": self.training_file,
            "feature_chip_size": self.chip_size,
            "output_chip_size": self.output_chip_size,
            "label_file": self.label_file,
            "label_bands": label_bands,
            "chip_from_raw_chip": self.chipper,
        }
        self.dataloader_kwargs = {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn,
        }

        self.test_loader = self._get_loader(
            Dataset(
                num_training_chips=self.test_samples,
                aoi=self.test_aoi,
                **self.dataset_kwargs,
            ),
            shuffle=False,
        )

        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()
        _optimizer = optim.SGD(
            self.net.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        self.optimizer = (
            hvd.DistributedOptimizer(_optimizer) if self.use_hvd else _optimizer
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._get_lr_lambda
        )

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value
        if self._is_root:
            print("Loss: %.4f" % value)
            self.run.log("Training Loss", value)

    @property
    def test_loss(self):
        return self._test_loss

    @test_loss.setter
    def test_loss(self, value):
        self._test_loss = value
        if self._is_root:
            print("Test loss: %.4f" % value)
            self.run.log("Test Loss", value)

    @property
    def epoch(self):
        """Current epoch"""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        if self._is_root:
            print("Epoch: ", value)
            self.run.log("Epoch", value)

    @property
    def lr(self):
        """Current learning rate"""
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
        if self._is_root:
            print("LR: %.4f" % value)
            self.run.log("lr", value)

    def _write_log(self, output_csv: str = "./outputs/log.csv") -> None:
        """Log current epoch, training and test loss, and learning rate to
        a csv file"""
        row = dict(
            epoch=self.epoch, loss=self.loss, test_loss=self.test_loss, lr=self.lr
        )
        self.log_df = self.log_df.append(row, ignore_index=True)
        self.log_df.to_csv(output_csv, index=False)

    def _get_lr_lambda(self, epoch: int) -> float:
        return self.learning_schedule_gamma ** epoch

    def _load_aoi(self, path, aoi_file: str) -> gpd.GeoDataFrame:
        if aoi_file is not None:
            aoi_file = os.path.join(path, aoi_file)
            return gpd.read_file(aoi_file)
        return None

    def _get_raster_specs(self, raster_file: str) -> tuple:
        with rio.open(raster_file) as src:
            num_bands = src.count
            res = src.res[0]

        return num_bands, res

    def _get_output_chip_size(self) -> int:
        # Not to be confused with "good" area
        test_chip = torch.Tensor(
            1, self.num_training_bands, self.chip_size, self.chip_size
        ).to(self.dev)
        return self.net.forward(test_chip).shape[2]

    def _get_training_ds(self) -> Dataset:
        """Modify the search area so samples are heavily focused on urban areas
        in early epochs"""
        if self.epoch > 10:
            aoi = self.test_aoi
        else:
            aoi = self.training_aoi

        # Scratch that, as it wasn't working w/ all bands (test outupt was nan)
        aoi = self.test_aoi

        return Dataset(
            num_training_chips=self.training_samples, aoi=aoi, **self.dataset_kwargs
        )

    def _get_loader(self, ds: Dataset, **kwargs) -> DataLoader:
        args = {**self.dataloader_kwargs, **kwargs}
        if self.use_hvd:

            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=hvd.size(), rank=hvd.rank()
            )
            # Can't use both shuffle and sampler args
            if "shuffle" in args.keys():
                args.pop("shuffle")
            args = {**args, **dict(sampler=sampler)}

        return DataLoader(ds, **args)

    @property
    def _is_root(self) -> bool:
        """Returns True if using a single GPU/CPU or if this instance is the
        "root" instance when using multiple GPUs"""
        return self.use_hvd and hvd.rank() == 0 or not self.use_hvd

    def train(self):
        if self.use_hvd:
            hvd.broadcast_parameters(self.net.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        for self.epoch in range(self.epochs):

            # Reset this at each epoch so samples change
            ds = self._get_training_ds()
            loader = self._get_loader(ds, shuffle=True)

            self.loss = self._step(loader) / self.training_samples

            with torch.no_grad():
                test_running_loss = self._step(loader=self.test_loader, training=False)
                self._write_swatches()

            self.test_loss = test_running_loss / self.test_samples
            self.lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            if self._is_root:
                self._write_log()
                if self.use_hvd:
                    self._merge_swatches()
                torch.save(self.net.state_dict(), "./outputs/model.pt")

    def _step(self, loader: DataLoader, training: bool = True) -> float:
        running_loss = 0.0
        self.net.train() if training else self.net.eval()

        for _, (_, data) in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(self.dev)
            labels = labels.to(self.dev)

            if training:
                self.optimizer.zero_grad()

            outputs = self.net(inputs.float())
            loss = self.criterion(outputs.squeeze(1), labels.squeeze().float())
            #            if np.isnan(loss.detach().cpu()):
            #                print(f"inputs {hvd.rank()}", inputs.float().detach().cpu().numpy())
            #                print(
            #                    f"outputs {hvd.rank()}", outputs.squeeze(1).detach().cpu().numpy()
            #                )
            #                print(f"labels {hvd.rank()}", labels.squeeze().detach().cpu().numpy())

            if training:
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item() * self.batch_size

        if self.use_hvd:
            running_loss = hvd.allreduce(
                torch.tensor(running_loss), name="avg_loss"
            ).item()

        inputs.detach()
        labels.detach()
        return running_loss

    def _write_swatches(self) -> None:
        self.net.eval()
        for i, _ in self.swatches.iterrows():
            swatch = self.swatches.loc[[i]]
            swatch_kwargs = self.dataset_kwargs.copy()
            swatch_kwargs.update({"aoi": swatch, "mode": "predict"})
            ds = Dataset(**swatch_kwargs)
            print(ds.num_chips)
            dl = self._get_loader(ds, batch_size=self.batch_size)
            with rio.open(self.label_file) as src:
                kwargs = src.meta.copy()
                out_array, transform = mask(src, [box(*ds.bounds)], crop=True)
            kwargs.update(
                {
                    "dtype": "float32",
                    "driver": "GTiff",
                    "bounds": ds.bounds,
                    "height": out_array.shape[1],
                    "width": out_array.shape[2],
                    "transform": transform,
                }
            )

            if self.use_hvd:
                output_file = f"outputs/swatch_{i}_{self.epoch}_{hvd.rank()}.tif"
            else:
                output_file = f"outputs/swatch_{i}_{self.epoch}.tif"
            swatch_np = np.empty(out_array.shape)
            for _, (ds_idxes, data) in enumerate(dl):
                output_tensor = self.net(data.to(self.dev).float())
                output_np = output_tensor.detach().cpu().numpy()
                for j, idx_tensor in enumerate(ds_idxes):
                    if len(output_np.shape) > 3:
                        prediction = output_np[j, :, 221:291, 221:291]
                    else:
                        prediction = output_np[:, 221:291, 221:291]

                    window = ds.get_cropped_window(
                        idx_tensor.detach().cpu().numpy(), 70, ds.aoi_transform
                    )
                    swatch_np[
                        :,
                        window.row_off : window.row_off + window.height,
                        window.col_off : window.col_off + window.width,
                    ] = prediction

            with rio.open(output_file, "w", **kwargs) as dst:
                dst.write(swatch_np)

            if not self.use_hvd:
                self._write_swatch_png(swatch_np, i)

    def _merge_swatches(self) -> None:
        for i, _ in self.swatches.iterrows():
            swatch_dss = [
                rio.open(os.path.join("outputs", f))
                for f in os.listdir("outputs")
                if f.startswith(f"swatch_{i}_{self.epoch}")
            ]
            output_path = f"outputs/swatch_{i}_{self.epoch}.tif"
            merge(swatch_dss, dst_path=output_path)
            swatch_np = rio.open(output_path).read()
            self._write_swatch_png(swatch_np, i)

    def _write_swatch_png(self, data: np.ndarray, swatch_idx: int) -> None:
        ramp = [
            (0, "#010101"),
            (1 / 3.0, "#ff0101"),
            (2 / 3.0, "#ffbb01"),
            (1, "#ffff01"),
        ]
        cm = LinearSegmentedColormap.from_list("urban", ramp)
        output_png = f"outputs/swatch_{swatch_idx}_{self.epoch}.png"
        # Save the first band (could do rgb if you wanted I think)
        rgb = Image.fromarray(np.uint8(cm(data[0, :, :]) * 255))
        rgb.save(output_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params-path", help="Path to params file, see src/model/configs"
    )
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = yaml.safe_load(f)

    if params["use_hvd"]:
        import horovod.torch as hvd

        hvd.init()

    for param in ["chipper", "criterion"]:
        if param in params.keys():
            params[param] = globals()[params[param]]

    run = Run.get_context()
    for k, v in params.items():
        run.tag(k, str(v))

    trainer = Trainer(**params)
    trainer.train()
