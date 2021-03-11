import math
import os
import random
import re

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from StreamingDataset import StreamingDataset
from Unet import Unet

cog_dir = 'data/cog/2016/training'
image_files = [
    os.path.join(cog_dir, f)
    for f in os.listdir(cog_dir)
    if not f.startswith('hm')
]

random.shuffle(image_files)
n_training_files = math.ceil(len(image_files) * 0.8)

training_files = image_files[:n_training_files]
tiles = [os.path.splitext(os.path.basename(f))[0] for f in training_files]
label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in tiles]

test_files = image_files[n_training_files:]
test_tiles = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
test_label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in test_tiles]


# Can go ~25 w/ zero padding, ~16 w/ reflect
BATCH_SIZE = 18
N_CHIPS_PER_TILE = 500
EPOCHS = 40
N_TILES = len(tiles)
N_SAMPLES_PER_EPOCH = N_CHIPS_PER_TILE * N_TILES
N_WORKERS = 8

N_TEST_TILES = len(test_files)
N_TEST_SAMPLES_PER_EPOCH = N_CHIPS_PER_TILE * N_TEST_TILES

szd = StreamingDataset(
    training_files,
    label_files,
    label_band=1,
    feature_chip_size=512,
    label_chip_size=70,
    num_chips_per_tile=N_CHIPS_PER_TILE
)

loader = DataLoader(
    szd,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    pin_memory=True
)

testsd = StreamingDataset(
        test_files,
        label_files,
        label_band=1,
        feature_chip_size=512,
        label_chip_size=70,
        num_chips_per_tile=N_CHIPS_PER_TILE
)

test_loader = DataLoader(
    testsd,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    pin_memory=True
)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

net = Unet(7)
net = net.float().to(dev)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2)

for epoch in range(EPOCHS):
    running_loss = 0.0
    test_running_loss = 0.0
    for i, data in enumerate(loader):
        print(i)
        net.train()
        inputs, labels = data
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()

        outputs = net(inputs.float())
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * BATCH_SIZE
    # Clear these off GPU so we can load up test data
    inputs.detach()
    labels.detach()
    print('Epoch %d loss: %.3f' %
          (epoch + 1, running_loss / N_SAMPLES_PER_EPOCH))

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            net.eval()
            inputs, labels = test_data
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            outputs = net(inputs.float())
            loss = criterion(outputs.squeeze(1), labels)
            test_running_loss += loss.item() * BATCH_SIZE

    print('Epoch %d test loss: %.3f' %
          (epoch + 1, test_running_loss / N_TEST_SAMPLES_PER_EPOCH))


torch.save(net.state_dict(), 'data/hall_model6.pt')
