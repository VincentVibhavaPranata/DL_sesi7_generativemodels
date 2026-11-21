# DL_sesi7_generativesmodel

Studi komparatif empiris model generatif pada **Fashion-MNIST**: Vanilla Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, dan DDPM, disertai eksperimen pembuktian batasan MLP Classifier untuk image generation.

Tugas Take-Home Deep Learning (D6421) Sesi 07, BINUS University.

## Penulis
- Nama: Vincent Vibhava Pranata
- NIM: ISI_NIM
- Kelas: ISI_KELAS

## Model
| Model | Loss utama | Catatan |
|---|---|---|
| Vanilla AE | MSE | 784-256-64-256-784, latent 64 |
| Sparse AE | MSE + L1 pada hidden layer (λ=1e-4) | penalti sparsity |
| VAE | BCE + KL divergence | reparameterization trick, latent 64 |
| Vanilla GAN | Adversarial (BCE, non-saturating) | generator dan diskriminator MLP, z=100 |
| DCGAN | Adversarial (BCE, non-saturating) | ConvTranspose/Conv + BatchNorm, z=100 |
| DDPM | MSE prediksi noise | UNet kecil, T=1000, jadwal noise linear |
| MLP Classifier | Cross-Entropy | 784-256-64-10, pembanding eksperimen batasan |

## Setup eksperimen (sama untuk semua model)
- Dataset Fashion-MNIST (70.000 citra, train+test digabung), split 80% train / 10% val / 10% test (56.000 / 7.000 / 7.000), seed 42
- Batch size 128, optimizer Adam
- Epoch: MLP 25, Vanilla/Sparse AE 30, VAE 30, Vanilla GAN 50, DCGAN 30, DDPM 30
- Metrik di test set: MSE, PSNR, SSIM (rekonstruksi), FID (5000 sampel, Inception-v3), latency per batch 64 sampel, waktu training per epoch

## Struktur repo
```
src/
  config.py          konfigurasi global (seed, batch size, latent, path)
  data.py            dataset dan split 80/10/10
  utils.py           checkpoint, state, loop training (fit)
  metrics.py         MSE, PSNR, SSIM, FID, latency
  vanilla_ae.py  sparse_ae.py  vae.py  gan.py  dcgan.py  ddpm.py  mlp_classifier.py
  mlp_limits.py      eksperimen pembuktian batasan MLP (Tahap 2-5) + Figur 3
  registry.py        memuat semua model terlatih
  train.py           latih model (CLI)
  evaluate.py        tabel metrik Bagian IV (CLI)
  make_figures.py    Figur 1, 2, 4, 5 (CLI)
figures/             figur hasil eksperimen
results/             tabel metrik (CSV/JSON)
```

## Cara menjalankan (Google Colab, GPU)
```python
!git clone https://github.com/VincentVibhavaPranata/DL_sesi7_generativesmodel.git
%cd DL_sesi7_generativesmodel
!pip -q install -r requirements.txt

# opsional: simpan checkpoint dan hasil ke Google Drive agar aman bila runtime putus
from google.colab import drive; drive.mount('/content/drive')
%env DL_OUT_DIR=/content/drive/MyDrive/DL_Sesi07

!python -m src.train --model all          # latih ketujuh model (atau --model vae, dst.)
!python -m src.mlp_limits                 # eksperimen batasan MLP Tahap 2-5 + Figur 3
!python -m src.evaluate                   # tabel metrik (FID DDPM paling lama)
!python -m src.make_figures               # Figur 1, 2, 4, 5
```
Uji cepat (data kecil, 1 epoch): tambahkan `--smoke` pada tiap perintah, dan `%env DL_DDPM_STEPS=20` agar sampling DDPM cepat. Jangan dipakai untuk hasil laporan.

## Hasil (satu run, seed 42, Google Colab GPU)
MLP Classifier: test accuracy **0.8909**.

| Model | MSE | PSNR (dB) | SSIM | FID | Latency 64 sampel (ms) | Waktu/epoch (s) |
|---|---|---|---|---|---|---|
| Vanilla AE | 0.0065 | 21.85 | 0.866 | - | 0.28 | 7.6 |
| Sparse AE | 0.0102 | 19.91 | 0.800 | - | 0.28 | 7.7 |
| VAE | 0.0142 | 18.49 | 0.763 | 62.39 | 0.15 | 8.4 |
| Vanilla GAN | - | - | - | 97.76 | 0.23 | 10.3 |
| DCGAN | - | - | - | 23.05 | 9.21 | 18.0 |
| DDPM | - | - | - | 20.93 | 16421 | 42.5 |

Eksperimen batasan MLP (latent 64 yang sama): decoder di atas fitur MLP yang dibekukan mencapai MSE 0.0181 / PSNR 17.42 / SSIM 0.707, dibanding Vanilla AE MSE 0.0065 / PSNR 21.85 / SSIM 0.866.

Catatan: latency AE = rekonstruksi (encode+decode), latency VAE = generasi (decode dari z acak). FID dihitung dengan Inception-v3 hasil foto natural sehingga bersifat relatif antar-model. Selisih FID DDPM dan DCGAN kecil dan berasal dari satu seed.

## Progress
- [x] Setup, dataset, helper metrik
- [x] MLP Classifier + eksperimen batasan (Tahap 1-5)
- [x] Vanilla AE, Sparse AE
- [x] VAE
- [x] Vanilla GAN, DCGAN
- [x] DDPM
- [x] Evaluasi dan figur
- [ ] Laporan PDF
