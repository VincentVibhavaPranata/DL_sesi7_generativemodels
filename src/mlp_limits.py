"""Eksperimen pembuktian batasan MLP Classifier (Tahap 2-5) + Figur 3.
Butuh checkpoint 'mlp_clf' dan 'VanillaAE' (jalankan src.train dulu).
    python -m src.mlp_limits
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import DEVICE, FIG_DIR, ensure_dirs, set_seed
from .data import get_data
from .metrics import eval_reconstruction
from .mlp_classifier import MLPClf, accuracy
from .utils import fit, load_ckpt, load_state, save_state
from .vanilla_ae import VanillaAE
from .viz import row, save_fig


class FrozenDec(nn.Module):
    """Tahap 4: decoder 64->256->784 di atas fitur MLP (z_MLP) yang dibekukan."""

    def __init__(s, clf):
        super().__init__()
        object.__setattr__(s, "clf", clf)   # sengaja tidak didaftarkan sebagai submodule (beku, tidak ikut state_dict)
        s.net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(s, x):
        with torch.no_grad():
            z = s.clf.features(x)
        return s.net(z).view(-1, 1, 28, 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    data = get_data(smoke=args.smoke)
    state = load_state()

    clf = MLPClf().to(DEVICE)
    load_ckpt(clf, "mlp_clf")
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    ae = VanillaAE().to(DEVICE)
    load_ckpt(ae, "VanillaAE")
    ae.eval()
    print("MLP test acc:", accuracy(clf, data.test_loader))

    xt, _ = next(iter(data.test_loader))
    xt = xt.to(DEVICE)
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight   # (256,784), (64,256), (10,64)

    with torch.no_grad():
        # Tahap 2: vektor one-hot kelas -> proyeksi balik W^T
        yc = torch.eye(10, device=DEVICE)
        x_s2 = torch.sigmoid(F.relu(F.relu(yc @ W3) @ W2) @ W1).view(-1, 1, 28, 28)
        # Tahap 3: fitur z_MLP citra uji -> proyeksi balik W^T
        z_t = clf.features(xt[:10])
        x_s3 = torch.sigmoid(F.relu(z_t @ W2) @ W1).view(-1, 1, 28, 28)

    # Tahap 4: frozen latent decoder dilatih dengan MSE
    ep = 1 if args.smoke else 20
    set_seed()
    dec4 = fit(FrozenDec(clf).to(DEVICE), lambda m, x: F.mse_loss(m(x), x),
               data.train_loader, ep, "FrozenDecoder", state)
    state["results"]["MLP_FrozenDecoder"] = eval_reconstruction(dec4, data.test_loader)
    state["results"].setdefault("VanillaAE", {}).update(eval_reconstruction(ae, data.test_loader))
    print("Frozen decoder:", state["results"]["MLP_FrozenDecoder"])
    print("Vanilla AE    :", state["results"]["VanillaAE"])

    # Tahap 5: sampling acak dari N(mu_z, diag(sigma_z^2)) lalu decode
    with torch.no_grad():
        Z = torch.cat([clf.features(x.to(DEVICE)) for x, _ in data.test_loader])
        mu_z, sd_z = Z.mean(0), Z.std(0)
        z_rand = mu_z + sd_z * torch.randn(16, 64, device=DEVICE)
        x_s5 = dec4.net(z_rand).view(-1, 1, 28, 28)

        rows = [(xt[:10], "Ground truth"),
                (x_s2, "Tahap 2: W^T dari one-hot kelas 0-9"),
                (x_s3, "Tahap 3: W^T dari z_MLP citra uji"),
                (dec4(xt[:10]), "Tahap 4: Frozen latent decoder (z_MLP)"),
                (ae(xt[:10]), "Pembanding: Vanilla AE (z_AE)"),
                (x_s5[:10], "Tahap 5: sampling N(mu_z, sigma_z^2)")]
    fig, axs = plt.subplots(len(rows), 1, figsize=(12, 1.6 * len(rows)))
    for ax, (im, t) in zip(axs, rows):
        row(ax, im, t)
    save_fig(fig, FIG_DIR / "fig3_mlp_limits.png")
    save_state(state)


if __name__ == "__main__":
    main()
