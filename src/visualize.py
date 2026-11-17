# Figur 1 (rekonstruksi), 2 (sampling 4x4), 4 (interpolasi), 5 (kurva loss + trajectory DDPM)
# Butuh (dari sel sebelumnya): ae, sae, vae, G1, G2, gen, xt, yt, histories, row,
#   to_01, ZD_GAN, DEVICE, PROJECT_DIR, set_seed, ddpm_sample, x64
import torch, matplotlib.pyplot as plt
from torchvision.utils import make_grid

# ---- Figur 1: grid rekonstruksi ----
idx = x64[:8]
with torch.no_grad():
    rows = [(idx, "Ground truth"), (ae(idx), "Vanilla AE"), (sae(idx), "Sparse AE"), (vae(idx), "VAE")]
fig, axs = plt.subplots(4, 1, figsize=(10, 6))
for ax, (im, t) in zip(axs, rows): row(ax, im, t)
plt.tight_layout(); plt.savefig(f"{PROJECT_DIR}/figures/fig1_recon.png", dpi=300); plt.show()

# ---- Figur 2: random sampling 4x4 ----
set_seed()
samples = {"A: VAE": gen["VAE"](16), "B: Vanilla GAN": gen["GAN"](16),
           "C: DCGAN": gen["DCGAN"](16), "D: DDPM": gen["DDPM"](16)}
fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
for ax, (k, v) in zip(axs, samples.items()):
    g = make_grid(v.detach().cpu().clamp(0, 1), nrow=4, padding=2)
    ax.imshow(g.permute(1, 2, 0).squeeze(), cmap="gray"); ax.set_title(k); ax.axis("off")
plt.tight_layout(); plt.savefig(f"{PROJECT_DIR}/figures/fig2_samples.png", dpi=300); plt.show()

# ---- Figur 4: interpolasi linear ----
alphas_i = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
def interp(zA, zB, decode): return torch.cat([decode((1-a)*zA + a*zB) for a in alphas_i])
with torch.no_grad():
    xA, xB = xt[yt == 1][:1], xt[yt == 9][:1]          # Trouser vs Ankle boot
    rows = [(interp(ae.encode(xA),  ae.encode(xB),  ae.decode),  "Vanilla AE"),
            (interp(sae.encode(xA), sae.encode(xB), sae.decode), "Sparse AE")]
    mA, _ = vae.encode(xA); mB, _ = vae.encode(xB)
    rows.append((interp(mA, mB, vae.decode), "VAE"))
    zA, zB = torch.randn(1, ZD_GAN, device=DEVICE), torch.randn(1, ZD_GAN, device=DEVICE)
    rows.append((interp(zA, zB, lambda z: to_01(G1(z))), "Vanilla GAN (z acak, tanpa encoder)"))
    rows.append((interp(zA, zB, lambda z: to_01(G2(z))), "DCGAN (z acak, tanpa encoder)"))
fig, axs = plt.subplots(5, 1, figsize=(8, 8))
for ax, (im, t) in zip(axs, rows): row(ax, im, t)
plt.tight_layout(); plt.savefig(f"{PROJECT_DIR}/figures/fig4_interp.png", dpi=300); plt.show()

# ---- Figur 5a: kurva loss enam model ----
fig, axs = plt.subplots(2, 3, figsize=(15, 7))
for ax, n in zip(axs.flat[:3], ["VanillaAE", "SparseAE", "VAE"]):
    ax.plot(histories[n]); ax.set_title(f"{n} loss"); ax.set_xlabel("epoch")
for ax, n in zip(axs.flat[3:5], ["GAN", "DCGAN"]):
    ax.plot(histories[n+"_G"], label="Generator"); ax.plot(histories[n+"_D"], label="Discriminator")
    ax.set_title(f"{n} loss"); ax.set_xlabel("epoch"); ax.legend()
axs.flat[5].plot(histories["DDPM"]); axs.flat[5].set_title("DDPM noise-prediction MSE"); axs.flat[5].set_xlabel("epoch")
plt.tight_layout(); plt.savefig(f"{PROJECT_DIR}/figures/fig5a_losses.png", dpi=300); plt.show()

# ---- Figur 5b: trajectory reverse diffusion x_T -> x_0 ----
set_seed()
_, traj = ddpm_sample(3, return_traj=True)
fig, axs = plt.subplots(3, 1, figsize=(12, 4.5))
for i, ax in enumerate(axs):
    imgs = to_01(torch.cat([s[i:i+1] for s in traj]).clamp(-1, 1))
    row(ax, imgs, f"Sampel {i+1}: x_T (kiri, noise) -> x_0 (kanan)")
plt.tight_layout(); plt.savefig(f"{PROJECT_DIR}/figures/fig5b_ddpm_traj.png", dpi=300); plt.show()
