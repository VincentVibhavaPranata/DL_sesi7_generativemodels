"""Konfigurasi global. Semua model WAJIB memakai konfigurasi yang sama agar perbandingan adil."""
import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
BATCH_SIZE = 128
LATENT_DIM = 64          # latent AE / Sparse AE / VAE
ZD_GAN = 100             # dimensi noise z untuk GAN dan DCGAN
IMG_SHAPE = (1, 28, 28)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Jumlah langkah difusi DDPM (default 1000, bisa dikecilkan lewat env var untuk smoke test)
T_DIFFUSION = int(os.environ.get("DL_DDPM_STEPS", 1000))

# Folder keluaran. Di Colab arahkan ke Google Drive agar tidak hilang saat runtime putus:
#   os.environ["DL_OUT_DIR"] = "/content/drive/MyDrive/DL_Sesi07"
OUT_DIR = Path(os.environ.get("DL_OUT_DIR", "."))
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
DATA_DIR = os.environ.get("DL_DATA_DIR", "./data")


def ensure_dirs():
    for d in (CKPT_DIR, FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
