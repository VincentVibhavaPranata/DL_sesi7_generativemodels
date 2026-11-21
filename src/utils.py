"""Helper umum: konversi range, checkpoint, state (histori loss, waktu, hasil), dan loop training."""
import json
import time

import numpy as np
import torch

from .config import CKPT_DIR, DEVICE, RES_DIR, ensure_dirs

to_pm1 = lambda x: x * 2 - 1        # [0,1]  -> [-1,1]
to_01 = lambda x: (x + 1) / 2       # [-1,1] -> [0,1]


# ---------- checkpoint ----------
def save_ckpt(model, name):
    ensure_dirs()
    torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")


def load_ckpt(model, name):
    model.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=DEVICE))
    return model


# ---------- state: histories, epoch_times, results ----------
STATE_PATH = RES_DIR / "state.json"


def load_state():
    if STATE_PATH.exists():
        s = json.load(open(STATE_PATH))
    else:
        s = {}
    for k in ("histories", "epoch_times", "results"):
        s.setdefault(k, {})
    return s


def save_state(state):
    ensure_dirs()
    json.dump(state, open(STATE_PATH, "w"), indent=2)


# ---------- loop training umum (AE, Sparse AE, VAE, DDPM, Frozen Decoder) ----------
def fit(model, loss_fn, loader, epochs, name, state, lr=1e-3):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    n_data = len(loader.dataset)
    hist, times = [], []
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            loss = loss_fn(model, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        times.append(time.time() - t0)
        hist.append(tot / n_data)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            print(f"{name} ep{ep + 1} loss {hist[-1]:.5f}")
    state["histories"][name] = hist
    state["epoch_times"][name] = float(np.mean(times))
    save_ckpt(model, name)
    model.eval()
    return model
