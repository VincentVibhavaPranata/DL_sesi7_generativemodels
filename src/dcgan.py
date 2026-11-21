"""DCGAN: generator ConvTranspose + discriminator Conv dengan BatchNorm.
Training memakai train_gan dan init_w dari gan.py."""
import torch.nn as nn

from .config import ZD_GAN


class DC_G(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.ConvTranspose2d(ZD_GAN, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),  # 7x7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),     # 14x14
            nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())                                            # 28x28

    def forward(s, z): return s.net(z.view(-1, ZD_GAN, 1, 1))


class DC_D(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),                                        # 14x14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),     # 7x7
            nn.Conv2d(128, 1, 7, 1, 0))

    def forward(s, x): return s.net(x).view(-1)
