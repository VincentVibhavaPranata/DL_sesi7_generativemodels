# Variational Autoencoder: encoder -> (mu, logvar) -> reparameterization -> decoder
# Loss = BCE (sum per gambar) + KL divergence ke N(0, I)
import torch, torch.nn as nn, torch.nn.functional as F

LATENT_DIM = 64

class VAE(nn.Module):
    def __init__(s, zd=LATENT_DIM):
        super().__init__()
        s.enc = nn.Sequential(nn.Flatten(), nn.Linear(784,512), nn.ReLU(), nn.Linear(512,256), nn.ReLU())
        s.mu, s.lv = nn.Linear(256,zd), nn.Linear(256,zd)
        s.dec = nn.Sequential(nn.Linear(zd,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(),
                              nn.Linear(512,784), nn.Sigmoid())
    def encode(s, x): h = s.enc(x); return s.mu(h), s.lv(h)
    def decode(s, z): return s.dec(z).view(-1,1,28,28)
    def forward(s, x): mu, _ = s.encode(x); return s.decode(mu)     # deterministik, buat evaluasi

def vae_loss(m, x):
    mu, lv = m.encode(x)
    z = mu + torch.randn_like(mu) * torch.exp(0.5*lv)               # reparameterization trick
    bce = F.binary_cross_entropy(m.decode(z), x, reduction="sum") / x.size(0)
    kl  = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.size(0)
    return bce + kl
