"""Helper visualisasi."""
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def row(ax, imgs, title):
    """Tampilkan sederet gambar (N,1,28,28) di [0,1] pada satu axis."""
    g = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=imgs.size(0), padding=2)
    ax.imshow(g.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def grid(ax, imgs, title, nrow=4):
    g = make_grid(imgs.detach().cpu().clamp(0, 1), nrow=nrow, padding=2)
    ax.imshow(g.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def save_fig(fig, path, dpi=300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("saved", path)
