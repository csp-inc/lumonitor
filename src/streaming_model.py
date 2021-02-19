import os
import re

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import xarray as xr

from StreamingDataset import StreamingDataset
from Unet import Unet

cog_dir = 'data/cog/2016'
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
    label_band='impervious',
    num_chips_per_tile=10000
)

loader = DataLoader(szd)

net = Unet(7)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
epochs = 1

for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(loader):
        print('hi')
        #print(data)
        # inputs, labels = data
        # optimizer.zero_grad()

#        outputs = net(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()
#        if i % 2000 == 1999:
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
