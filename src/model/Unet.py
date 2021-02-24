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

def decoder_block(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=(2, 2),
        stride=(2, 2),
        padding=0
    )

def norm_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        conv_block(in_channels, out_channels)
    )

class Unet(torch.nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.encoder_block_0 = conv_block(self.in_channels, 32)
        self.encoder_block_1 = conv_block(32, 64)
        self.encoder_block_2 = conv_block(64, 128)
        self.encoder_block_3 = conv_block(128, 256)
        self.encoder_block_4 = conv_block(256, 512)
        self.center = conv_block(512, 1024)
        self.decoder_block_4 = decoder_block(1024, 512)
        self.post_decoder_4 = norm_relu(1024, 512)
        self.decoder_block_3 = decoder_block(512, 256)
        self.post_decoder_3 = norm_relu(512, 256)
        self.decoder_block_2 = decoder_block(256, 128)
        self.post_decoder_2 = norm_relu(256, 128)
        self.decoder_block_1 = decoder_block(128, 64)
        self.post_decoder_1 = norm_relu(128, 64)
        self.decoder_block_0 = decoder_block(64, 32)
        self.post_decoder_0 = norm_relu(64, 32)
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_0 = self.encoder_block_0(x)
        x = nn.MaxPool2d(2)(encoder_0)
        encoder_1 = self.encoder_block_1(x)
        x = nn.MaxPool2d(2)(encoder_1)
        encoder_2 = self.encoder_block_2(x)
        x = nn.MaxPool2d(2)(encoder_2)
        encoder_3 = self.encoder_block_3(x)
        x = nn.MaxPool2d(2)(encoder_3)
        encoder_4 = self.encoder_block_4(x)
        x = nn.MaxPool2d(2)(encoder_4)
        x = self.center(x)
        decoder_4 = self.decoder_block_4(x)
        x = self.post_decoder_4(torch.cat((decoder_4, encoder_4), dim=1))
        decoder_3 = self.decoder_block_3(x)
        x = self.post_decoder_3(torch.cat((decoder_3, encoder_3), dim=1))
        decoder_2 = self.decoder_block_2(x)
        x = self.post_decoder_2(torch.cat((decoder_2, encoder_2), dim=1))
        decoder_1 = self.decoder_block_1(x)
        x = self.post_decoder_1(torch.cat((decoder_1, encoder_1), dim=1))
        decoder_0 = self.decoder_block_0(x)
        x = self.post_decoder_0(torch.cat((decoder_0, encoder_0), dim=1))
        return self.output(x).squeeze(0)
