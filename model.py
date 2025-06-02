import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


# One resnet block (with identity)
# bn -> con -> relu
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate=0.3, identity=True, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, out_planes)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = conv3x3(out_planes, out_planes, stride=stride)

        # "identity" convolution
        self.identity = nn.Sequential()
        if identity:
            self.identity = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=stride),
            )

    def forward(self, x):
        x_org = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x += self.identity(x_org)

        return x


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, widen_factor=4):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 32.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(
            in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(
            D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module(
                'shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with nested skip pathways and deep supervision.
    """
    def __init__(self, in_channels=1, out_channels=1, deep_supervision=False, features=None, block=VGGBlock):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv_00 = block(in_channels, features[0])
        self.conv_10 = block(features[0], features[1])
        self.conv_20 = block(features[1], features[2])
        self.conv_30 = block(features[2], features[3])
        self.conv_40 = block(features[3], features[4])

        # Decoder
        self.conv_01 = block(features[0] + features[1], features[0])
        self.conv_11 = block(features[1] + features[2], features[1])
        self.conv_21 = block(features[2] + features[3], features[2])
        self.conv_31 = block(features[3] + features[4], features[3])

        self.conv_02 = block(features[0] * 2 + features[1], features[0])
        self.conv_12 = block(features[1] * 2 + features[2], features[1])
        self.conv_22 = block(features[2] * 2 + features[3], features[2])

        self.conv_03 = block(features[0] * 3 + features[1], features[0])
        self.conv_13 = block(features[1] * 3 + features[2], features[1])

        self.conv_04 = block(features[0] * 4 + features[1], features[0])

        # Deep Supervision
        self.final_conv1 = nn.Conv2d(features[0], out_channels, 1)
        self.final_conv2 = nn.Conv2d(features[0], out_channels, 1)
        self.final_conv3 = nn.Conv2d(features[0], out_channels, 1)
        self.final_conv4 = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x, l=4):
        # Encoder
        x_00 = self.conv_00(x)
        x_10 = self.conv_10(self.pool(x_00))
        x_20 = self.conv_20(self.pool(x_10))
        x_30 = self.conv_30(self.pool(x_20))
        x_40 = self.conv_40(self.pool(x_30))

        # Decoder
        x_01 = self.conv_01(torch.cat((x_00, self.up(x_10)), 1))
        if l == 1:
            return self.final_conv1(x_01)
        x_11 = self.conv_11(torch.cat((x_10, self.up(x_20)), 1))
        x_21 = self.conv_21(torch.cat((x_20, self.up(x_30)), 1))
        x_31 = self.conv_31(torch.cat((x_30, self.up(x_40)), 1))

        x_02 = self.conv_02(torch.cat((x_00, x_01, self.up(x_11)), 1))
        if l == 2:
            return self.final_conv2(x_02)
        x_12 = self.conv_12(torch.cat((x_10, x_11, self.up(x_21)), 1))
        x_22 = self.conv_22(torch.cat((x_20, x_21, self.up(x_31)), 1))

        x_03 = self.conv_03(torch.cat((x_00, x_01, x_02, self.up(x_12)), 1))
        if l == 3:
            return self.final_conv3(x_03)
        x_13 = self.conv_13(torch.cat((x_10, x_11, x_12, self.up(x_22)), 1))

        x_04 = self.conv_04(
            torch.cat((x_00, x_01, x_02, x_03, self.up(x_13)), 1))

        # Deep Supervision
        if self.deep_supervision:
            x_ds1 = self.final_conv1(x_01)
            x_ds2 = self.final_conv2(x_02)
            x_ds3 = self.final_conv3(x_03)
            x_ds4 = self.final_conv4(x_04)
            return x_ds1, x_ds2, x_ds3, x_ds4

        return self.final_conv4(x_04)


class Unet(nn.Module):
    """
    U-Net architecture with symmetric skip connections for image-to-image tasks.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.deep_supervision = False
        
        # Building the encoder (downsampling path)
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(VGGBlock(prev_channels, feature))
            prev_channels = feature

        # Building the decoder (upsampling path)
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(VGGBlock(feature * 2, feature))

        # Bottleneck layer
        self.bottleneck = VGGBlock(features[-1], features[-1] * 2)

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward pass
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck forward pass
        x = self.bottleneck(x)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsampling
            skip_connection = skip_connections[idx // 2]

            # Resize x if its shape differs from the skip connection
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            # Concatenate along the channel dimension
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)  # Apply double convolution

        return self.final_conv(x)
