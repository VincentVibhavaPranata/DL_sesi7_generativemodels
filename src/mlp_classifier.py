"""MLP Classifier 784-256-64-10 (Tahap 1 eksperimen pembuktian). Target test acc >= 88%."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, set_seed
from .utils import load_ckpt, save_ckpt


class MLPClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, 256), nn.Linear(256, 64), nn.Linear(64, 10)

    def features(self, x):
        h1 = F.relu(self.fc1(x.view(x.size(0), -1)))
        return F.relu(self.fc2(h1))          # z_MLP (64)

    def forward(self, x):
        return self.fc3(self.features(x))


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    c = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return c / n


def train_mlp(data, state, epochs=25, lr=1e-3):
    """Latih MLP, simpan checkpoint dengan val acc terbaik, catat test acc ke state."""
    set_seed()
    clf = MLPClf().to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    best_val = 0.0
    for ep in range(epochs):
        clf.train()
        tot = 0.0
        for x, y in data.train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        va = accuracy(clf, data.val_loader)
        if va > best_val:
            best_val = va
            save_ckpt(clf, "mlp_clf")
        print(f"ep{ep + 1} loss {tot / len(data.train_loader.dataset):.4f} val_acc {va:.4f}")
    load_ckpt(clf, "mlp_clf")
    clf.eval()
    test_acc = accuracy(clf, data.test_loader)
    print("TEST ACC:", test_acc)
    state["results"]["MLP_Classifier"] = {"test_acc": test_acc}
    return clf
