# DL_sesi7_generativemodels

Studi komparatif empiris model generatif pada **Fashion-MNIST**: Vanilla Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, dan DDPM, disertai eksperimen pembuktian batasan MLP Classifier untuk image generation.

Tugas Take-Home Deep Learning (D6421) Sesi 07, BINUS University.

## Penulis
- Nama: Vincent Vibhava Pranata
- NIM: 2802415523

## Isi Proyek
| Model | Loss utama | Catatan |
|---|---|---|
| Vanilla AE | MSE | latent 64 |
| Sparse AE | MSE + L1 pada hidden layer | penalti sparsity |
| VAE | BCE + KL divergence | reparameterization trick |
| Vanilla GAN | Adversarial (BCE) | arsitektur MLP |
| DCGAN | Adversarial (BCE) | arsitektur konvolusional |
| DDPM | MSE prediksi noise | UNet, T=1000 |
| MLP Classifier | Cross-Entropy | pembanding, 784-256-64-10 |

## Setup Eksperimen (sama untuk semua model)
- Dataset: Fashion-MNIST (70.000 citra), split 80% train / 10% val / 10% test
- Batch size 128, random seed 42
- Metrik di test set: MSE, PSNR, SSIM, FID, training time per epoch, latency per batch (64 sampel)

## Struktur Repo
```
src/        kode PyTorch modular per model
figures/    figur hasil eksperimen
results/    tabel metrik (CSV/JSON)
```

## Cara Menjalankan
1. Buka Google Colab, pilih Runtime GPU (T4)
2. Clone repo ini dan install dependensi:
```
git clone https://github.com/VincentVibhavaPranata/DL_sesi7_generativesmodel.git
cd DL_sesi7_generativesmodel
pip install -r requirements.txt
```
3. Jalankan skrip di `src/` sesuai urutan: `common.py` -> `mlp_classifier.py` -> model AE -> `vae.py` -> `gan.py` / `dcgan.py` -> `ddpm.py` -> `evaluate.py`

## Progress
- [ ] Setup, dataset, helper metrik
- [ ] MLP Classifier + eksperimen batasan (tahap 1-5)
- [ ] Vanilla AE, Sparse AE
- [ ] VAE
- [ ] Vanilla GAN, DCGAN
- [ ] DDPM
- [ ] Evaluasi dan figur
