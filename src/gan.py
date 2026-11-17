# Vanilla GAN (MLP) + fungsi training adversarial yang dipakai GAN dan DCGAN
# Butuh: DEVICE, train_loader, train_ds, histories, epoch_times, save_ckpt, set_seed, to_pm1
import time, numpy as np, torch, torch.nn as nn

ZD_GAN = 100

class GAN_G(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(ZD_GAN,256), nn.LeakyReLU(0.2), nn.Linear(256,512), nn.LeakyReLU(0.2),
                              nn.Linear(512,1024), nn.LeakyReLU(0.2), nn.Linear(1024,784), nn.Tanh())
    def forward(s, z): return s.net(z).view(-1,1,28,28)

class GAN_D(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(784,1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                              nn.Linear(1024,512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                              nn.Linear(512,256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256,1))
    def forward(s, x): return s.net(x).view(-1)

def init_w(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.normal_(m.weight, 0, 0.02)
    elif isinstance(m, nn.BatchNorm2d): nn.init.normal_(m.weight, 1, 0.02); nn.init.zeros_(m.bias)

def train_gan(G, D, epochs, name):
    set_seed()
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); hG, hD, times = [], [], []
    for ep in range(epochs):
        G.train(); D.train(); t0 = time.time(); sg = sd = 0
        for x, _ in train_loader:
            x = to_pm1(x.to(DEVICE)); b = x.size(0)
            one, zero = torch.ones(b, device=DEVICE), torch.zeros(b, device=DEVICE)
            fake = G(torch.randn(b, ZD_GAN, device=DEVICE))
            lD = bce(D(x), one) + bce(D(fake.detach()), zero)       # update Discriminator
            oD.zero_grad(); lD.backward(); oD.step()
            lG = bce(D(fake), one)                                  # update Generator (non-saturating)
            oG.zero_grad(); lG.backward(); oG.step()
            sg += lG.item()*b; sd += lD.item()*b
        times.append(time.time()-t0); hG.append(sg/len(train_ds)); hD.append(sd/len(train_ds))
        if (ep+1) % 10 == 0: print(f"{name} ep{ep+1} G {hG[-1]:.3f} D {hD[-1]:.3f}")
    histories[name+"_G"], histories[name+"_D"], epoch_times[name] = hG, hD, float(np.mean(times))
    save_ckpt(G, name+"_G"); G.eval(); D.eval(); return G
