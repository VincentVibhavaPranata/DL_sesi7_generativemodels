"""Latih model. Contoh:
    python -m src.train --model all
    python -m src.train --model vae
    python -m src.train --model all --smoke      # uji cepat (data kecil, 1 epoch)
"""
import argparse

import torch.nn.functional as F  # noqa: F401  (dipakai lambda loss bila perlu)

from .config import DEVICE, ZD_GAN, ensure_dirs, set_seed
from .data import get_data
from .dcgan import DC_D, DC_G
from .ddpm import UNet, ddpm_loss
from .gan import GAN_D, GAN_G, init_w, train_gan
from .mlp_classifier import train_mlp
from .sparse_ae import SparseAE, sae_loss
from .utils import fit, load_state, save_state
from .vae import VAE, vae_loss
from .vanilla_ae import VanillaAE, ae_loss

STAGES = ["mlp", "vanilla_ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"]
EPOCHS = {"mlp": 25, "vanilla_ae": 30, "sparse_ae": 30, "vae": 30, "gan": 50, "dcgan": 30, "ddpm": 30}


def run_stage(stage, state, smoke):
    data = get_data(smoke=smoke)          # loader baru per model -> urutan shuffle konsisten
    ep = 1 if smoke else EPOCHS[stage]
    set_seed()
    if stage == "mlp":
        train_mlp(data, state, epochs=ep)
    elif stage == "vanilla_ae":
        fit(VanillaAE().to(DEVICE), ae_loss, data.train_loader, ep, "VanillaAE", state)
    elif stage == "sparse_ae":
        fit(SparseAE().to(DEVICE), sae_loss, data.train_loader, ep, "SparseAE", state)
    elif stage == "vae":
        fit(VAE().to(DEVICE), vae_loss, data.train_loader, ep, "VAE", state)
    elif stage == "gan":
        train_gan(GAN_G().to(DEVICE), GAN_D().to(DEVICE), data.train_loader, ep, "GAN", state)
    elif stage == "dcgan":
        G, D = DC_G().to(DEVICE), DC_D().to(DEVICE)
        G.apply(init_w); D.apply(init_w)
        train_gan(G, D, data.train_loader, ep, "DCGAN", state)
    elif stage == "ddpm":
        fit(UNet().to(DEVICE), ddpm_loss, data.train_loader, ep, "DDPM", state, lr=2e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=STAGES + ["all"])
    ap.add_argument("--smoke", action="store_true", help="uji cepat: subset data kecil, 1 epoch")
    args = ap.parse_args()

    ensure_dirs()
    state = load_state()
    for stage in (STAGES if args.model == "all" else [args.model]):
        print(f"\n=== {stage} ===")
        run_stage(stage, state, args.smoke)
        save_state(state)          # simpan tiap model selesai (aman bila runtime putus)


if __name__ == "__main__":
    main()
