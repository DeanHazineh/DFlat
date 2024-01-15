import torch


def cart_grid(gsize, gdx, radial_symmetry, dtype=torch.float32, device="cpu"):
    if radial_symmetry:
        nr = gsize[-1] // 2 + 1
        x, y = torch.meshgrid(
            torch.arange(0, nr, dtype=dtype),
            torch.arange(0, 1, dtype=dtype),
            indexing="xy",
        )
    else:
        x, y = torch.meshgrid(
            torch.arange(0, gsize[-1], dtype=dtype),
            torch.arange(0, gsize[-2], dtype=dtype),
            indexing="xy",
        )
        x = x - (x.shape[-1] - 1) / 2
        y = y - (y.shape[-2] - 1) / 2

    x = x * gdx[-1]
    y = y * gdx[-2]
    return x.to(device=device), y.to(device=device)
