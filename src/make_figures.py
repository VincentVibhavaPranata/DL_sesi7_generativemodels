"""Figur 1, 2, 4, 5a, 5b dari checkpoint + state. Figur 3 dihasilkan oleh src.mlp_limits.
    python -m src.make_figures
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import DEVICE, FIG_DIR, ZD_GAN, ensure_dirs, set_seed
from .data import get_data
from .ddpm import ddpm_sample
from .registry import load_models, sample_fns
from .utils import load_state, to_01
from .viz import grid, row, save_fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    data = get_data(smoke=args.smoke)
    hist = load_state()["histories"]
    m = load_models()
    gen = sample_fns(m)
    xt, yt = next(iter(data.test_loader))
    xt, yt = xt.to(DEVICE), yt.to(DEVICE)
    ae, sae, vae, G1, G2 = m["ae"], m["sae"], m["vae"], m["G1"], m["G2"]

    # ---- Figur 1: grid rekonstruksi (8 sampel) ----
    idx = xt[:8]
    with torch.no_grad():
        rows = [(idx, "Ground truth"), (ae(idx), "Vanilla AE"), (sae(idx), "Sparse AE"), (vae(idx), "VAE")]
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    for ax, (im, t) in zip(axs, rows):
        row(ax, im, t)
    save_fig(fig, FIG_DIR / "fig1_recon.png")

    # ---- Figur 2: random sampling 4x4 ----
    set_seed()
    samples = {"A: VAE": gen["VAE"](16), "B: Vanilla GAN": gen["GAN"](16),
               "C: DCGAN": gen["DCGAN"](16), "D: DDPM": gen["DDPM"](16)}
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, (k, v) in zip(axs, samples.items()):
        grid(ax, v, k)
    save_fig(fig, FIG_DIR / "fig2_samples.png")

    # ---- Figur 4: interpolasi linear (alpha = 0, 0.2, ..., 1.0) ----
    alphas_i = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def interp(zA, zB, decode):
        return torch.cat([decode((1 - a) * zA + a * zB) for a in alphas_i])

    with torch.no_grad():
        xA, xB = xt[yt == 1][:1], xt[yt == 9][:1]          # Trouser vs Ankle boot
        if len(xA) == 0 or len(xB) == 0:                   # cadangan bila kelas tidak ada di batch pertama
            xA, xB = xt[:1], xt[1:2]
        rows = [(interp(ae.encode(xA), ae.encode(xB), ae.decode), "Vanilla AE"),
                (interp(sae.encode(xA), sae.encode(xB), sae.decode), "Sparse AE")]
        mA, _ = vae.encode(xA)
        mB, _ = vae.encode(xB)
        rows.append((interp(mA, mB, vae.decode), "VAE"))
        zA, zB = torch.randn(1, ZD_GAN, device=DEVICE), torch.randn(1, ZD_GAN, device=DEVICE)
        rows.append((interp(zA, zB, lambda z: to_01(G1(z))), "Vanilla GAN (z acak, tanpa encoder)"))
        rows.append((interp(zA, zB, lambda z: to_01(G2(z))), "DCGAN (z acak, tanpa encoder)"))
    fig, axs = plt.subplots(5, 1, figsize=(8, 8))
    for ax, (im, t) in zip(axs, rows):
        row(ax, im, t)
    save_fig(fig, FIG_DIR / "fig4_interp.png")

    # ---- Figur 5a: kurva training loss enam model ----
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    for ax, n in zip(axs.flat[:3], ["VanillaAE", "SparseAE", "VAE"]):
        ax.plot(hist[n]); ax.set_title(f"{n} loss"); ax.set_xlabel("epoch")
    for ax, n in zip(axs.flat[3:5], ["GAN", "DCGAN"]):
        ax.plot(hist[n + "_G"], label="Generator"); ax.plot(hist[n + "_D"], label="Discriminator")
        ax.set_title(f"{n} loss"); ax.set_xlabel("epoch"); ax.legend()
    axs.flat[5].plot(hist["DDPM"]); axs.flat[5].set_title("DDPM noise-prediction MSE")
    axs.flat[5].set_xlabel("epoch")
    save_fig(fig, FIG_DIR / "fig5a_losses.png")

    # ---- Figur 5b: trajectory reverse diffusion x_T -> x_0 ----
    set_seed()
    _, traj = ddpm_sample(m["unet"], 3, return_traj=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 4.5))
    for i, ax in enumerate(axs):
        imgs = to_01(torch.cat([s[i:i + 1] for s in traj]).clamp(-1, 1))
        row(ax, imgs, f"Sampel {i + 1}: x_T (kiri, noise) -> x_0 (kanan)")
    save_fig(fig, FIG_DIR / "fig5b_ddpm_traj.png")


if __name__ == "__main__":
    main()
