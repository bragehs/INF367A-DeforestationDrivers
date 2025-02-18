import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        size = x.size()[2:]
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.avg_pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EnhancedSatelliteSegmentationModel(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super(EnhancedSatelliteSegmentationModel, self).__init__()
        
        # Initial Conv Block
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder path with residual blocks
        self.enc1 = ResidualBlock(64, 64)
        self.enc2 = ResidualBlock(64, 128, stride=2)
        self.enc3 = ResidualBlock(128, 256, stride=2)
        self.enc4 = ResidualBlock(256, 512, stride=2)
        
        # ASPP module
        self.aspp = ASPP(512, 256)
        
        # Decoder path with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Conv2d(256, 256, kernel_size=1)
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Conv2d(64, 64, kernel_size=1)
        
        # Additional upsampling to match input resolution
        self.final_up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final_up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        # Store input size for potential interpolation
        input_size = x.size()[2:]
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Encoder path with skip connections
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        x = self.enc4(skip3)
        
        # ASPP module
        x = self.aspp(x)
        
        # Decoder path with skip connections
        x = self.dec1(x)
        x = x + self.skip1(skip3)
        x = self.dropout(x)
        
        x = self.dec2(x)
        x = x + self.skip2(skip2)
        x = self.dropout(x)
        
        x = self.dec3(x)
        x = x + self.skip3(skip1)
        x = self.dropout(x)
        
        # Additional upsampling to match input resolution
        x = self.final_up1(x)
        x = self.final_up2(x)
        x = self.final_conv(x)
        
        # Optional: force output to match input spatial dimensions exactly
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
            
        return x

