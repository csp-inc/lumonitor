import os
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

tiles = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in tiles]

szd = StreamingDataset(
    image_files,
    label_files,
    label_band=1,
    num_chips_per_tile=500
)

BATCH_SIZE = 25

loader = DataLoader(
    szd,
    num_workers=2,
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
optimizer = optim.SGD(net.parameters(), lr=0.01)
epochs = 8

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()

        outputs = net(inputs.float())
        # ????
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # ????
        if i % BATCH_SIZE == BATCH_SIZE - 1: 
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / BATCH_SIZE))
            running_loss = 0.0

torch.save(net.state_dict(), 'data/hall_model3.pt')
