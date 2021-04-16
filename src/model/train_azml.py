import math
import os
import random
import re

from azureml.core import Run
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.MosaicDataset import MosaicDataset as Dataset
from models.Unet_padded import Unet

training_file = 'model/conus_hls_median_2016.vrt'
label_file = '/vsiaz/hls/NLCD_2016_Impervious_L48_20190405.tif'

BATCH_SIZE = 1
CHIP_SIZE = 512
EPOCHS = 50
N_SAMPLES_PER_EPOCH = 1000
N_TEST_SAMPLES_PER_EPOCH = 100

N_WORKERS = 6

N_BANDS = 7
net = Unet(N_BANDS)

LABEL_BAND = 1

test_chip = torch.Tensor(1, N_BANDS, CHIP_SIZE, CHIP_SIZE)
OUTPUT_CHIP_SIZE = net.forward(test_chip).shape[2]

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


szd = Dataset(
    training_file,
    label_file,
    label_band=LABEL_BAND,
    feature_chip_size=CHIP_SIZE,
    label_chip_size=OUTPUT_CHIP_SIZE,
    num_chips=N_SAMPLES_PER_EPOCH
)

loader = DataLoader(
    szd,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
    shuffle=True
)

testsd = Dataset(
    training_file,
    label_file,
    label_band=LABEL_BAND,
    feature_chip_size=CHIP_SIZE,
    label_chip_size=OUTPUT_CHIP_SIZE,
    num_chips=N_TEST_SAMPLES_PER_EPOCH
)

test_loader = DataLoader(
    testsd,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
    shuffle=False
)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

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
    # Clear these off GPU so we can load up test data
    inputs.detach()
    labels.detach()
    print('Epoch %d loss: %.4f' %
          (epoch + 1, running_loss / N_SAMPLES_PER_EPOCH))

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            net.eval()
            inputs, labels = test_data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            outputs = net(inputs.float())
            loss = criterion(outputs.squeeze(1), labels.float())
            test_running_loss += loss.item() * BATCH_SIZE

    print('Epoch %d test loss: %.4f' %
          (epoch + 1, test_running_loss / N_TEST_SAMPLES_PER_EPOCH))
    print('LR: %.4f' % optimizer.param_groups[0]["lr"])
    scheduler.step()


torch.save(net.state_dict(), 'data/imp_model_padded_2.pt')
