import argparse
import math
import os
import random
import re
import yaml

from azureml.core import Run
import geopandas as gpd
import numpy as np
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

    szd = Dataset(
        num_training_chips=training_samples,
        **dataset_kwargs
    )
    loader = DataLoader(szd, shuffle=True, **dataloader_kwargs)

    testsd = Dataset(
        num_training_chips=test_samples,
        **dataset_kwargs
    )
    test_loader = DataLoader(testsd, shuffle=False, **dataloader_kwargs)

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
        torch.save(net.state_dict(), './outputs/{run.id}_{i}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params-path',
        help="Path to params file, see src/model/configs"
    )
    args = parser.parse_args()

    with open(args.params_path) as f:
        params = yaml.load(f)

    run = Run.get_context()
    [run.tag(k, str(v)) for k, v in params.items()]

    train(**params)
