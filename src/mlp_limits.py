# Eksperimen pembuktian batasan MLP Classifier (Tahap 2-5)
# Butuh: clf (beku), ae, DEVICE, test_loader, fit, eval_reconstruction, set_seed, results (dari sel sebelumnya)
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def run_mlp_limits(clf, ae, xt, yt):
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight   # (256,784), (64,256), (10,64)
    with torch.no_grad():
        # Tahap 2: one-hot kelas -> W^T
        yc = torch.eye(10, device=DEVICE)
        x_s2 = torch.sigmoid(F.relu(F.relu(yc @ W3) @ W2) @ W1).view(-1,1,28,28)
        # Tahap 3: z_MLP citra uji -> W^T
        z_t = clf.features(xt[:10])
        x_s3 = torch.sigmoid(F.relu(z_t @ W2) @ W1).view(-1,1,28,28)
    return x_s2, x_s3

class FrozenDec(nn.Module):
    # Tahap 4: decoder 64->256->784 di atas fitur MLP yang dibekukan
    def __init__(s, clf):
        super().__init__()
        s.clf = clf
        s.net = nn.Sequential(nn.Linear(64,256), nn.ReLU(), nn.Linear(256,784), nn.Sigmoid())
    def forward(s, x):
        with torch.no_grad(): z = s.clf.features(x)
        return s.net(z).view(-1,1,28,28)

def sample_latent(dec4, clf, test_loader, n=16):
    # Tahap 5: sampling dari N(mu_z, diag(sigma_z^2)) lalu decode
    with torch.no_grad():
        Z = torch.cat([clf.features(x.to(DEVICE)) for x, _ in test_loader])
        mu_z, sd_z = Z.mean(0), Z.std(0)
        z_rand = mu_z + sd_z * torch.randn(n, 64, device=DEVICE)
        return dec4.net(z_rand).view(-1,1,28,28)

def row(ax, imgs, title):
    g = make_grid(imgs.detach().cpu().clamp(0,1), nrow=imgs.size(0), padding=2)
    ax.imshow(g.permute(1,2,0).squeeze(), cmap="gray"); ax.set_title(title, fontsize=9); ax.axis("off")
