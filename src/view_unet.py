import torch
from torchinfo import summary

from Unet import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Unet(7)
model = model.to(device)

# check keras-like model summary using torchsummary
summary(model, input_size=(16, 7, 256, 256), device="cpu")

