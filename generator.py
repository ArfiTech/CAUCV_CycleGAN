import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_filters=64, num_resnet_blocks=9):
        super(Generator, self).__init__()

        # 인코더 모델
        self.encoder = self.build_encoder(input_channels, num_filters)

        # 변환 모델
        self.transform = self.build_transform(num_filters, num_resnet_blocks)

        # 디코더 모델
        self.decoder = self.build_decoder(num_filters, output_channels)

    def build_encoder(self, input_channels, num_filters):
        encoder = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filters*4),
            nn.ReLU(inplace=True)
        )
        return encoder

    def build_transform(self, num_filters, num_resnet_blocks):
        # ResNet 블록 생성
        resnet_blocks = []
        for _ in range(num_resnet_blocks):
            resnet_blocks += [ResnetBlock(num_filters*4)]

        transform = nn.Sequential(
            *resnet_blocks
        )
        return transform

    def build_decoder(self, num_filters, output_channels):
        decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        return decoder

    def forward(self, x):
        # 인코더
        enc_features = self.encoder(x)

        # 변환 모델
        transformed_features = self.transform(enc_features)

        # 디코더
        output = self.decoder(transformed_features)

        return output


class ResnetBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResnetBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_filters)
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out = out + residual
        return out
