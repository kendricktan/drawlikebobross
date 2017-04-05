import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset

from torch.autograd import Variable

# Picture dimension: 450 x 337
# Resize to 256 * 256


# Custom weight initialization
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Encoder (Image -> z)
class Encoder(nn.Module):
    def __init__(self, z_dim, h_dim=128, filter_num=64, channel_num=3):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            # Input: (channel_num) x 256 x 256
            nn.Conv2d(channel_num, filter_num, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (filter_num) x 128 x 128
            nn.Conv2d(filter_num, filter_num * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (filter_num * 2) x 64 x 64
            nn.Conv2d(filter_num * 2, filter_num * 2, 8, 4, 1, bias=False),
            nn.BatchNorm2d(filter_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (filter_num * 2) x 16 x 16
            nn.Conv2d(filter_num * 2, filter_num, 8, 4, 1, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.LeakyReLU(0.2, inplace=True),
            # (filter_num) x 3 x 3
        )

        self.fc = nn.Sequential(
            # x.size(0) * -1
            nn.Linear(filter_num * 3 * 3, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )

        self.z_dim = z_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Decoder (z -> Image)
class Decoder(nn.Module):
    def __init__(self, z_dim, filter_num=64, channel_num=3):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            # z -> upsampling -> image
            nn.ConvTranspose2d(z_dim, filter_num * 8, 8, 1, 0, bias=False),
            nn.BatchNorm2d(filter_num * 8),
            nn.ReLU(True),
            # (filter_num * 8) * 16 x 16
            nn.ConvTranspose2d(filter_num * 8, filter_num * \
                               4, 8, 4, 2, bias=False),
            nn.BatchNorm2d(filter_num * 4),
            nn.ReLU(True),
            # (filter_num * 4) * 32 x 32
            nn.ConvTranspose2d(filter_num * 4, filter_num * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_num * 2),
            nn.ReLU(True),
            # (filter_num * 2) * 64 x 64
            nn.ConvTranspose2d(filter_num * 2, filter_num,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(True),
            # (filter_num) * 128 * 128
            nn.ConvTranspose2d(filter_num, channel_num, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # channel_num * 256 * 256
        )

        self.z_dim = z_dim

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv(x)
        return x


# Discriminator (Z -> REAL/FAKE image)
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            # x.size(0) * -1
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# G(A) = B
class Generator(nn.Module):
    def __init__(self, z_dim, h_dim=128, filter_num=64, channel_num=3):
        super(Generator, self).__init__()

        encoder = Encoder(z_dim, h_dim, filter_num, channel_num)
        decoder = Decoder(z_dim, filter_num, channel_num)

        self.conv = encoder.conv
        self.fc = encoder.fc

        self.conv_t = decoder.conv

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv_t(x)
        return x
