# Vanilla Autoencoder: 784 -> 256 -> 64 (latent) -> 256 -> 784, loss MSE
import torch.nn as nn, torch.nn.functional as F

LATENT_DIM = 64

class VanillaAE(nn.Module):
    def __init__(s, zd=LATENT_DIM):
        super().__init__()
        s.enc = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,zd))
        s.dec = nn.Sequential(nn.Linear(zd,256), nn.ReLU(), nn.Linear(256,784), nn.Sigmoid())
    def encode(s, x): return s.enc(x)
    def decode(s, z): return s.dec(z).view(-1,1,28,28)
    def forward(s, x): return s.decode(s.encode(x))

def ae_loss(m, x): return F.mse_loss(m(x), x)
