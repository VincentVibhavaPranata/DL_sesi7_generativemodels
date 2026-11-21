"""Evaluasi semua model di test set -> tabel Bagian IV (results/results.json dan results/table_bagian_IV.csv).
Butuh semua checkpoint (jalankan src.train dan src.mlp_limits dulu).
    python -m src.evaluate                 # FID 5000 sampel (DDPM paling lama)
    python -m src.evaluate --fid-samples 2000
    python -m src.evaluate --skip-fid
"""
import argparse
import json

import pandas as pd
import torch

from .config import DEVICE, LATENT_DIM, RES_DIR, ZD_GAN, ensure_dirs, set_seed
from .data import get_data
from .dcgan import DC_D
from .ddpm import ddpm_sample
from .gan import GAN_D
from .metrics import compute_fid, count_params, eval_reconstruction, measure_latency
from .mlp_classifier import accuracy
from .registry import load_models, sample_fns
from .utils import load_state, save_state, to_01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid-samples", type=int, default=5000)
    ap.add_argument("--skip-fid", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    data = get_data(smoke=args.smoke)
    state = load_state()
    res = state["results"]
    m = load_models()
    gen = sample_fns(m)

    res["MLP_Classifier"] = {"test_acc": accuracy(m["clf"], data.test_loader)}

    x64 = next(iter(data.test_loader))[0][:64].to(DEVICE)
    z64 = torch.randn(64, LATENT_DIM, device=DEVICE)
    zg64 = torch.randn(64, ZD_GAN, device=DEVICE)

    # Rekonstruksi (MSE/PSNR/SSIM) + latency rekonstruksi 64 gambar untuk keluarga AE
    for name, model in [("VanillaAE", m["ae"]), ("SparseAE", m["sae"]), ("VAE", m["vae"])]:
        res.setdefault(name, {}).update(eval_reconstruction(model, data.test_loader))
        res[name]["latency_ms"] = measure_latency(lambda model=model: model(x64))
    # VAE: latency yang dilaporkan = generasi (decode dari z acak), BUKAN rekonstruksi
    res["VAE"]["latency_ms"] = measure_latency(lambda: m["vae"].decode(z64))

    # Latency generator (64 sampel)
    res.setdefault("GAN", {})["latency_ms"] = measure_latency(lambda: m["G1"](zg64))
    res.setdefault("DCGAN", {})["latency_ms"] = measure_latency(lambda: m["G2"](zg64))
    res.setdefault("DDPM", {})["latency_ms"] = measure_latency(lambda: ddpm_sample(m["unet"], 64),
                                                             n_runs=2, warmup=1)

    # FID (jumlah sampel SAMA untuk semua model)
    if not args.skip_fid:
        for name in ["VAE", "GAN", "DCGAN", "DDPM"]:
            res[name]["FID"] = compute_fid(gen[name], data.test_loader, n_samples=args.fid_samples)
            print(name, "FID", res[name]["FID"])

    # Jumlah parameter (GAN/DCGAN: generator, dan diskriminator terpisah)
    for name, model in [("VanillaAE", m["ae"]), ("SparseAE", m["sae"]), ("VAE", m["vae"]),
                        ("GAN", m["G1"]), ("DCGAN", m["G2"]), ("DDPM", m["unet"])]:
        res[name]["params"] = count_params(model)
    res["GAN"]["params_D"] = count_params(GAN_D())
    res["DCGAN"]["params_D"] = count_params(DC_D())

    # Waktu training per epoch (dari state hasil src.train)
    for name in ["VanillaAE", "SparseAE", "VAE", "GAN", "DCGAN", "DDPM"]:
        if name in state["epoch_times"]:
            res[name]["epoch_s"] = state["epoch_times"][name]

    save_state(state)
    json.dump(res, open(RES_DIR / "results.json", "w"), indent=2)
    df = pd.DataFrame(res).T
    df.to_csv(RES_DIR / "table_bagian_IV.csv")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
