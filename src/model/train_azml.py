import argparse
import math
import os
import random
import re
import yaml

from azureml.core import Run
import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import rasterio as rio
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(
        epochs,
        training_samples,
        test_samples,
#        seed,
#        node_count=1,
        learning_rate=0.01,
        learning_schedule_gamma=0.975,
        momentum=0.8,
        chip_size=512,
        aoi_file='conus.geojson',
        batch_size=8,
        num_workers=6
):
    run = Run.get_context()
    offline = run._run_id.startswith("OfflineRun")
    path = 'data/azml' if offline else 'model/data/azml'

    training_file = os.path.join(path, 'conus_hls_median_2016.vrt')
    label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'
    aoi_file = os.path.join(path, aoi_file)
    aoi = gpd.read_file(aoi_file)

    with rio.open(training_file) as src:
        num_bands = src.count

    with rio.open(label_file) as dst:
        dst_kwargs = dst.meta.copy()
    dst_kwargs.update({'count': 1, 'dtype': 'float32'})

    net = Unet(num_bands)

    test_chip = torch.Tensor(1, num_bands, chip_size, chip_size)
    output_chip_size = net.forward(test_chip).shape[2]

    dataset_kwargs = {
        'feature_file': training_file,
        'feature_chip_size': chip_size,
        'output_chip_size': output_chip_size,
        'aoi': aoi,
        'label_file': label_file
    }

    dataloader_kwargs = {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'pin_memory': True,
        'worker_init_fn': worker_init_fn
    }

    testsd = Dataset(
        num_training_chips=test_samples,
        **dataset_kwargs
    )
    test_loader = DataLoader(testsd, shuffle=False, **dataloader_kwargs)

    def write_swatches(
            net: Unet,
            swatches: gpd.GeoDataFrame,
            epoch: int
    ) -> None:
        ramp = [
            (0, '#010101'),
            (1/3., '#ff0101'),
            (2/3., '#ffbb01'),
            (1, '#ffff01')
        ]
        cm = LinearSegmentedColormap.from_list("urban", ramp)
        for swatch in swatches:
            swatch_kwargs = dataset_kwargs.copy()
            swatch_kwargs.update({'aoi': swatch})
            ds = Dataset(**swatch_kwargs)
            dl = DataLoader(ds, **dataloader_kwargs)
            output_file = f'outputs/swatch_{swatch.name}_{epoch}.png'
            for i, data in enumerate(dl):
                output_tensor = net(data.float().to(dev))
                output_np = output_tensor.detach().cpu().numpy()
                prediction = output_np[: 221:291, 221:291]
                window = ds.get_cropped_window(i, output_chip_size)
                # Figure out how to create this of the correct size
                # I think what we need to do is get the resolution
                # from the input raster, and the xmax and xmin from the
                # ds, then ceiling(range / res)
                output_np[
                    window.col_off:window.col_off + window.width,
                    window.row_off:window.row_off + window.height
                ] = prediction
                rgb = Image.fromarray(np.uint8(cm(output_np)*255))
                rgb.save(output_file)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', dev)
    net = net.float().to(dev)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    lambda1 = lambda epoch: learning_schedule_gamma ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    for epoch in range(epochs):
        np.random.seed()

        # Reset this at each epoch so samples change
        szd = Dataset(
            num_training_chips=training_samples,
            **dataset_kwargs
        )
        loader = DataLoader(szd, shuffle=True, **dataloader_kwargs)

        running_loss = 0.0
        test_running_loss = 0.0
        for i, data in enumerate(loader):
            net.train()
            inputs, labels = data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size

        inputs.detach()
        labels.detach()
        train_loss = running_loss / training_samples
        print('Epoch %d loss: %.4f' % (epoch + 1, train_loss))

        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                net.eval()
                inputs, labels = test_data
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                outputs = net(inputs.float())
                loss = criterion(outputs.squeeze(1), labels.float())
                test_running_loss += loss.item() * batch_size

            write_swatches(net, swatches, swatch_dl, epoch)

#            with rio.open(output_file
#            for i, test_area_data in enumerate(test_area_loader):
#                output_torch = net(data.float().to(dev))
#                output_np = output_torch.detach().cpu().numpy()
#                prediction = output_np[:, 221:291, 221:291]
#                window = test_area_ds.get_cropped_window(i, output_chip_size)


        test_loss = test_running_loss / test_samples
        print('Epoch %d test loss: %.4f' % (epoch + 1, test_loss))
        run.log_row("Loss", x=epoch+1, Training=train_loss, Test=test_loss)

        lr = optimizer.param_groups[0]["lr"]
        print('LR: %.4f' % lr)
        run.log("lr", lr)
        scheduler.step()

        # Just for testing, we dont' need to save GB of models
        # ^^ Well then how do we go back?
        # Only save the last one?
        torch.save(net.state_dict(), f'./outputs/model.pt')


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
    [run.tag(k, str(v)) for k, v in params.items()]

    train(**params)
