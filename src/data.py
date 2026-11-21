"""Dataset Fashion-MNIST: gabung 70k citra, split 80/10/10 dengan seed tetap."""
from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_DIR, SEED


@dataclass
class Data:
    train_ds: object
    val_ds: object
    test_ds: object
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def get_data(batch_size: int = BATCH_SIZE, num_workers: int = 2, smoke: bool = False) -> Data:
    """Pixel dalam range [0, 1]. Model berbasis Tanh (GAN/DCGAN/DDPM) memakai utils.to_pm1."""
    tfm = transforms.ToTensor()
    train_raw = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_raw = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
    full = ConcatDataset([train_raw, test_raw])

    n = len(full)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED)
    )

    if smoke:  # subset kecil, hanya untuk uji cepat
        train_ds = Subset(train_ds, range(512))
        val_ds = Subset(val_ds, range(256))
        test_ds = Subset(test_ds, range(256))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return Data(train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)
