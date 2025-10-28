import torch
import torch.nn as nn


class Deeper3DUnetWithDropout(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.3):
        super().__init__()

        def _conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
                nn.Conv3d(out_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
            )

        self.enc1, self.pool1 = _conv_block(in_channels, 64), nn.MaxPool3d(2)
        self.enc2, self.pool2 = _conv_block(64, 128), nn.MaxPool3d(2)
        self.enc3, self.pool3 = _conv_block(128, 256), nn.MaxPool3d(2)
        self.enc4, self.pool4 = _conv_block(256, 512), nn.MaxPool3d(2)
        self.bottleneck = _conv_block(512, 1024)

        self.upconv4, self.dec4 = nn.ConvTranspose3d(1024, 512, 2, 2), _conv_block(
            1024, 512
        )
        self.upconv3, self.dec3 = nn.ConvTranspose3d(512, 256, 2, 2), _conv_block(
            512, 256
        )
        self.upconv2, self.dec2 = nn.ConvTranspose3d(256, 128, 2, 2), _conv_block(
            256, 128
        )
        self.upconv1, self.dec1 = nn.ConvTranspose3d(128, 64, 2, 2), _conv_block(
            128, 64
        )

        self.final_conv = nn.Conv3d(64, out_channels, 1)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.dropout(self.pool1(e1))
        e2 = self.enc2(p1)
        p2 = self.dropout(self.pool2(e2))
        e3 = self.enc3(p2)
        p3 = self.dropout(self.pool3(e3))
        e4 = self.enc4(p3)
        p4 = self.dropout(self.pool4(e4))
        b = self.bottleneck(p4)
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], 1))
        return self.final_conv(d1)
