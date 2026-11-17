# Sparse Autoencoder: sama seperti Vanilla AE, tapi hidden layer (256) kena penalti L1
import torch.nn as nn, torch.nn.functional as F

LATENT_DIM = 64
LAMBDA = 1e-4   # kekuatan sparsity

class SparseAE(nn.Module):
    def __init__(s, zd=LATENT_DIM):
        super().__init__()
        s.enc1 = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU())
        s.enc2 = nn.Linear(256, zd)
        s.dec  = nn.Sequential(nn.Linear(zd,256), nn.ReLU(), nn.Linear(256,784), nn.Sigmoid())
    def encode(s, x): return s.enc2(s.enc1(x))
    def decode(s, z): return s.dec(z).view(-1,1,28,28)
    def forward(s, x): return s.decode(s.encode(x))

def sae_loss(m, x):
    h = m.enc1(x); xh = m.decode(m.enc2(h))
    return F.mse_loss(xh, x) + LAMBDA * h.abs().sum(1).mean()
