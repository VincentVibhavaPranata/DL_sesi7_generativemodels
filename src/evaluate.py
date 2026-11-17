# Evaluasi semua model di test set: MSE/PSNR/SSIM, FID, latency, waktu training per epoch
# Butuh (dari sel sebelumnya): ae, sae, vae, G1, G2, ddpm_sample, eval_reconstruction,
#   compute_fid, measure_latency, test_loader, results, epoch_times, DEVICE, LATENT_DIM, ZD_GAN, to_01
import pandas as pd, torch

x64  = next(iter(test_loader))[0][:64].to(DEVICE)
z64  = torch.randn(64, LATENT_DIM, device=DEVICE)
zg64 = torch.randn(64, ZD_GAN, device=DEVICE)

gen = {"VAE":   lambda n: vae.decode(torch.randn(n, LATENT_DIM, device=DEVICE)),
       "GAN":   lambda n: to_01(G1(torch.randn(n, ZD_GAN, device=DEVICE))),
       "DCGAN": lambda n: to_01(G2(torch.randn(n, ZD_GAN, device=DEVICE))),
       "DDPM":  lambda n: to_01(ddpm_sample(n).clamp(-1, 1))}

# Rekonstruksi + latency (AE family: latency rekonstruksi 64 gambar)
for n, m in [("VanillaAE", ae), ("SparseAE", sae), ("VAE", vae)]:
    results.setdefault(n, {}).update(eval_reconstruction(m))
    results[n]["latency_ms"] = measure_latency(lambda m=m: m(x64))
results["VAE"]["latency_ms"] = measure_latency(lambda: vae.decode(z64))   # VAE: latency generasi

# Latency generator
results.setdefault("GAN", {})["latency_ms"]   = measure_latency(lambda: G1(zg64))
results.setdefault("DCGAN", {})["latency_ms"] = measure_latency(lambda: G2(zg64))
results.setdefault("DDPM", {})["latency_ms"]  = measure_latency(lambda: ddpm_sample(64), n_runs=2, warmup=1)

# FID (5000 sampel, sama untuk semua model)
for n in ["VAE", "GAN", "DCGAN", "DDPM"]:
    results[n]["FID"] = compute_fid(gen[n], n_samples=5000)

# Training time per epoch
for n in ["VanillaAE", "SparseAE", "VAE", "GAN", "DCGAN", "DDPM"]:
    results[n]["epoch_s"] = epoch_times[n]
