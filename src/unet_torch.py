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
    pooled_encoder = nn.MaxPool2d(encoder)
    return pooled_encoder, encoder

def decoder_block(in_channels, out_channels, concatter, x):
    decoder = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=(2,2),
            stride=(2,2),
            padding=1)(x)
    decoder = torch.cat(decoder, concatter)
    decoder = nn.BatchNorm2d(out_channels)(decoder)
    decoder = nn.ReLU(inplace=True)(decoder)
    return conv_block(???, out_channels)(decoder)

class Unet(nn.Module):
    def __init__(self, in_channels):
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)

    def forward(self, x):
        pooled_encoder0, encoder0 = encoder_block(self.in_channels, 32, x)
        pooled_encoder1, encoder1 = encoder_block(32, 64, pooled_encoder0)
        pooled_encoder2, encoder2 = encoder_block(64, 128, pooled_encoder1)
        pooled_encoder3, encoder3 = encoder_block(128, 256, pooled_encoder2)
        pooled_encoder4, encoder4 = encoder_block(256, 512, pooled_encoder3)
        # In the GEE implementation this was just another conv block
        center = self.upsample(pooled_encoder4)

        decoder4 = decoder_block(center, encoder4, 512)  # 16
        decoder3 = decoder_block(decoder4, encoder3, 256)  # 32
        decoder2 = decoder_block(decoder3, encoder2, 128)  # 64
        decoder1 = decoder_block(decoder2, encoder1, 64)  # 128
        decoder0 = decoder_block(decoder1, encoder0, 32)  # 256
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

