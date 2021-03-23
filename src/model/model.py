import math
import os
import random

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#from SerialStreamingDataset import SerialStreamingDataset
from StreamingDataset import StreamingDataset
#from LuDataset import LuDataset
from Unet import Unet

def get_data_loaders(train_batch_size, val_batch_size):
    cog_dir = 'data/cog/2016/training'

    image_files = [
        os.path.join(cog_dir, f)
        for f in os.listdir(cog_dir)
        if not f.startswith('hm')
    ]

    random.shuffle(image_files)

    training_files = [
        os.path.join(cog_dir, '11SLT.tif'),
        os.path.join(cog_dir, '11SMT.tif'),
        os.path.join(cog_dir, '10SGJ.tif'),
    ]

    #training_files = image_files[:n_training_files]
    tiles = [os.path.splitext(os.path.basename(f))[0] for f in training_files]
    label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in tiles]

    #test_files = image_files[n_training_files:]
    #test_files = [f for f in image_files if f not in training_files]
    test_files = training_files
    test_tiles = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
    test_label_files = [os.path.join(cog_dir, 'hm_' + t + '.tif') for t in test_tiles]

    # Can go ~25 w/ zero padding, ~16 w/ reflect
    N_CHIPS_PER_TILE = 100
    N_WORKERS = 7
    LABEL_BAND=6

    szd = StreamingDataset(
        training_files,
        label_files,
        label_band=LABEL_BAND,
        feature_chip_size=512,
        label_chip_size=70,
        num_chips_per_tile=N_CHIPS_PER_TILE
    )

    train_loader = DataLoader(
        szd,
        num_workers=N_WORKERS,
        batch_size=train_batch_size,
        pin_memory=True,
    )

    testsd = StreamingDataset(
        test_files,
        test_label_files,
        label_band=LABEL_BAND,
        feature_chip_size=512,
        label_chip_size=70,
        num_chips_per_tile=N_CHIPS_PER_TILE,
    )

    val_loader = DataLoader(
        testsd,
        num_workers=N_WORKERS,
        batch_size=val_batch_size,
        pin_memory=True
    )

    return train_loader, val_loader


net = Unet(7)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

net = net.to(dev)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
batch_size = 17
log_interval = 50
train_loader, val_loader = get_data_loaders(batch_size, batch_size)

trainer = create_supervised_trainer(net, optimizer, criterion, device=dev)

val_metrics = {"mse": Loss(criterion)}
evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=dev)


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.4f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['mse']:.4f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {metrics['mse']:.4f}")

trainer.run(train_loader, max_epochs=100)
