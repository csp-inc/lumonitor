import math
import os
import random
import re

from azureml.core import Run
import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet

run = Run.get_context()
offline = run._run_id.startswith("OfflineRun")
path = 'data/azml' if offline else 'model/data/azml'

training_file = os.path.join(path, 'conus_hls_median_2016.vrt')
aoi_file = os.path.join(path, 'conus.geojson')

label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'

BATCH_SIZE = 8
CHIP_SIZE = 512
EPOCHS = 5 # 50
#N_SAMPLES_PER_EPOCH = 10000
#N_TEST_SAMPLES_PER_EPOCH = 2500
N_SAMPLES_PER_EPOCH = 10
N_TEST_SAMPLES_PER_EPOCH = 2

N_WORKERS = 6

N_BANDS = 7
net = Unet(N_BANDS)

LABEL_BAND = 1

test_chip = torch.Tensor(1, N_BANDS, CHIP_SIZE, CHIP_SIZE)
OUTPUT_CHIP_SIZE = net.forward(test_chip).shape[2]


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

aoi = gpd.read_file(aoi_file)

dataset_kwargs = {
    'feature_file': training_file,
    'feature_chip_size': CHIP_SIZE,
    'output_chip_size': OUTPUT_CHIP_SIZE,
    'aoi': aoi,
    'label_file': label_file
}

dataloader_kwargs = {
    'num_workers': N_WORKERS,
    'batch_size': BATCH_SIZE,
    'pin_memory': True,
    'worker_init_fn': worker_init_fn
}

szd = Dataset(
    num_training_chips=N_SAMPLES_PER_EPOCH,
    **dataset_kwargs
)
loader = DataLoader(szd, shuffle=True, **dataloader_kwargs)

testsd = Dataset(
    num_training_chips=N_TEST_SAMPLES_PER_EPOCH,
    **dataset_kwargs
)
test_loader = DataLoader(testsd, shuffle=False, **dataloader_kwargs)


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)

net = net.float().to(dev)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
lambda1 = lambda epoch: 0.975 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

for epoch in range(EPOCHS):
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
        running_loss += loss.item() * BATCH_SIZE

    inputs.detach()
    labels.detach()
    train_loss = running_loss / N_SAMPLES_PER_EPOCH
    print('Epoch %d loss: %.4f' % (epoch + 1, train_loss))

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            net.eval()
            inputs, labels = test_data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            outputs = net(inputs.float())
            loss = criterion(outputs.squeeze(1), labels.float())
            test_running_loss += loss.item() * BATCH_SIZE

    test_loss = test_running_loss / N_TEST_SAMPLES_PER_EPOCH
    print('Epoch %d test loss: %.4f' % (epoch + 1, test_loss))
    run.log_row("Loss", x=epoch+1, Training=train_loss, Test=test_loss)

    lr = optimizer.param_groups[0]["lr"]
    print('LR: %.4f' % lr)
    run.log("lr", lr)
    scheduler.step()


torch.save(net.state_dict(), './outputs/imp_model_padded_2.pt')
