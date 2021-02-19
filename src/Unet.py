import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def encoder_block(in_channels, out_channels, x):
    encoder = conv_block(in_channels, out_channels)(x)
    pooled_encoder = nn.MaxPool2d(2)(encoder)
    return pooled_encoder, encoder

def decoder_block(in_channels, out_channels, x, concatter):
    decoder = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=0
    )(x)
    decoder = torch.cat((decoder, concatter), dim=1)
    decoder = nn.BatchNorm2d(in_channels)(decoder)
    decoder = nn.ReLU(inplace=True)(decoder)
    return conv_block(in_channels, out_channels)(decoder)

class Unet(torch.nn.Module):
    def __init__(self, x):
        super(Unet, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        pooled_encoder0, encoder0 = encoder_block(self.in_channels, 32, x)
        pooled_encoder1, encoder1 = encoder_block(32, 64, pooled_encoder0)
        pooled_encoder2, encoder2 = encoder_block(64, 128, pooled_encoder1)
        pooled_encoder3, encoder3 = encoder_block(128, 256, pooled_encoder2)
        pooled_encoder4, encoder4 = encoder_block(256, 512, pooled_encoder3)
        center = conv_block(512, 1024)(pooled_encoder4)
        decoder4 = decoder_block(1024, 512, center, encoder4)
        decoder3 = decoder_block(512, 256, decoder4, encoder3)
        decoder2 = decoder_block(256, 128, decoder3, encoder2)
        decoder1 = decoder_block(128, 64, decoder2, encoder1)
        decoder0 = decoder_block(64, 32, decoder1, encoder0)

        output = nn.Sequential(
            nn.Conv2d(32, 1, (1, 1)),
            nn.Sigmoid()
        )(decoder0)

        return output
