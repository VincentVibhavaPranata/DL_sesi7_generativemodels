"""Memuat semua model terlatih dari checkpoint dan menyediakan fungsi sampling seragam."""
import torch

from .config import DEVICE, LATENT_DIM, ZD_GAN
from .dcgan import DC_G
from .ddpm import UNet, ddpm_sample
from .gan import GAN_G
from .mlp_classifier import MLPClf
from .sparse_ae import SparseAE
from .utils import load_ckpt, to_01
from .vae import VAE
from .vanilla_ae import VanillaAE


def _load(model, name):
    model = model.to(DEVICE)
    load_ckpt(model, name)
    model.eval()
    return model


def load_models():
    m = {
        "clf": _load(MLPClf(), "mlp_clf"),
        "ae": _load(VanillaAE(), "VanillaAE"),
        "sae": _load(SparseAE(), "SparseAE"),
        "vae": _load(VAE(), "VAE"),
        "G1": _load(GAN_G(), "GAN_G"),
        "G2": _load(DC_G(), "DCGAN_G"),
        "unet": _load(UNet(), "DDPM"),
    }
    for p in m["clf"].parameters():
        p.requires_grad = False
    return m


def sample_fns(m):
    """Fungsi n -> n gambar sintetis di range [0, 1] untuk keempat model generatif."""
    return {
        "VAE": lambda n: m["vae"].decode(torch.randn(n, LATENT_DIM, device=DEVICE)),
        "GAN": lambda n: to_01(m["G1"](torch.randn(n, ZD_GAN, device=DEVICE))),
        "DCGAN": lambda n: to_01(m["G2"](torch.randn(n, ZD_GAN, device=DEVICE))),
        "DDPM": lambda n: to_01(ddpm_sample(m["unet"], n).clamp(-1, 1)),
    }
