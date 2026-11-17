# Fungsi training umum untuk model AE (dipakai juga oleh VAE & DDPM)
# Butuh: DEVICE, train_loader, histories, epoch_times, save_ckpt (dari common)
import time, numpy as np, torch

def fit(model, loss_fn, epochs, name, lr=1e-3):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist, times = [], []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0
        for x, _ in train_loader:
            x = x.to(DEVICE); loss = loss_fn(model, x)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*x.size(0)
        times.append(time.time()-t0); hist.append(tot/len(train_loader.dataset))
        if (ep+1) % 5 == 0: print(f"{name} ep{ep+1} loss {hist[-1]:.5f}")
    histories[name], epoch_times[name] = hist, float(np.mean(times))
    save_ckpt(model, name); model.eval(); return model
