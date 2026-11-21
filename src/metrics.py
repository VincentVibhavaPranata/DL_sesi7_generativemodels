"""Metrik evaluasi: MSE, PSNR, SSIM (rekonstruksi), FID (generasi), latency, jumlah parameter."""
import time

import torch

from .config import DEVICE


@torch.no_grad()
def eval_reconstruction(recon_fn, loader):
    """recon_fn: batch x (B,1,28,28) di [0,1] -> rekonstruksi di [0,1]. Dihitung di test set."""
    from torchmetrics import MeanSquaredError
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    mse = MeanSquaredError().to(DEVICE)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    for x, _ in loader:
        x = x.to(DEVICE)
        xh = recon_fn(x).clamp(0, 1)
        mse.update(xh, x)
        psnr.update(xh, x)
        ssim.update(xh, x)
    return {"MSE": mse.compute().item(), "PSNR": psnr.compute().item(), "SSIM": ssim.compute().item()}


@torch.no_grad()
def compute_fid(sample_fn, real_loader, n_samples=5000, gen_bs=250):
    """sample_fn(n) -> n gambar sintetis (n,1,28,28) di [0,1]. Real = gambar test set.
    Gunakan n_samples yang SAMA untuk semua model."""
    # Impor di dalam fungsi: butuh torch-fidelity terpasang SEBELUM torchmetrics pertama kali dipakai.
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    to3 = lambda t: t.clamp(0, 1).repeat(1, 3, 1, 1)  # grayscale -> 3 channel

    seen = 0
    for x, _ in real_loader:
        x = x.to(DEVICE)[: n_samples - seen]
        fid.update(to3(x), real=True)
        seen += x.size(0)
        if seen >= n_samples:
            break

    made = 0
    while made < n_samples:
        b = min(gen_bs, n_samples - made)
        fid.update(to3(sample_fn(b).to(DEVICE)), real=False)
        made += b
    return fid.compute().item()


@torch.no_grad()
def measure_latency(fn, n_runs=50, warmup=10):
    """Rata-rata waktu (ms) satu kali panggil fn()."""
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
