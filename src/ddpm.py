# DDPM: forward diffusion (noise schedule linear, T=1000) + UNet prediksi noise + reverse sampling
# Butuh: DEVICE, to_pm1 (dari common), dan variabel global `unet` (model terlatih) saat sampling
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

T = 1000
betas  = torch.linspace(1e-4, 0.02, T, device=DEVICE)
alphas = 1 - betas
abar   = torch.cumprod(alphas, 0)

def t_embed(t, dim):
    half = dim // 2
    f = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)

class Res(nn.Module):
    def __init__(s, i, o, td):
        super().__init__()
        s.n1, s.c1 = nn.GroupNorm(8, i), nn.Conv2d(i, o, 3, padding=1)
        s.t = nn.Linear(td, o)
        s.n2, s.c2 = nn.GroupNorm(8, o), nn.Conv2d(o, o, 3, padding=1)
        s.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x, te):
        h = s.c1(F.silu(s.n1(x))) + s.t(te)[:, :, None, None]
        return s.c2(F.silu(s.n2(h))) + s.sk(x)

class UNet(nn.Module):
    def __init__(s, ch=64, td=256):
        super().__init__()
        s.ch = ch
        s.temb = nn.Sequential(nn.Linear(ch, td), nn.SiLU(), nn.Linear(td, td))
        s.inc = nn.Conv2d(1, ch, 3, padding=1)
        s.d1, s.dn1 = Res(ch, ch, td),     nn.Conv2d(ch, ch, 3, 2, 1)         # 28 -> 14
        s.d2, s.dn2 = Res(ch, 2*ch, td),   nn.Conv2d(2*ch, 2*ch, 3, 2, 1)     # 14 -> 7
        s.mid = Res(2*ch, 2*ch, td)
        s.up2, s.u2 = nn.ConvTranspose2d(2*ch, 2*ch, 4, 2, 1), Res(4*ch, ch, td)
        s.up1, s.u1 = nn.ConvTranspose2d(ch, ch, 4, 2, 1),     Res(2*ch, ch, td)
        s.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))
    def forward(s, x, t):
        te = s.temb(t_embed(t, s.ch))
        h1 = s.d1(s.inc(x), te)
        h2 = s.d2(s.dn1(h1), te)
        h  = s.mid(s.dn2(h2), te)
        h  = s.u2(torch.cat([s.up2(h), h2], 1), te)
        h  = s.u1(torch.cat([s.up1(h), h1], 1), te)
        return s.out(h)

def ddpm_loss(m, x):
    x = to_pm1(x); t = torch.randint(0, T, (x.size(0),), device=DEVICE)
    eps = torch.randn_like(x); a = abar[t][:, None, None, None]
    return F.mse_loss(m(a.sqrt()*x + (1-a).sqrt()*eps, t), eps)       # prediksi noise

@torch.no_grad()
def ddpm_sample(n, return_traj=False):
    x = torch.randn(n, 1, 28, 28, device=DEVICE); traj = [x.clone()]
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps = unet(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
        if return_traj and t % 100 == 0: traj.append(x.clone())
    return (x, traj) if return_traj else x
