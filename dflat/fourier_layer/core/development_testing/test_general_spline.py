import torch
from dflat.fourier_layer.core.ops_transform_util import (
    radial_2d_transform,
    tf_generalSpline_regular1DGrid,
)

if __name__ == "__main__":
    # Simple test demonstration about the accuracy of our implementation
    # The code matches the pyhank official release
    import matplotlib.pyplot as plt
    import scipy.special
    import numpy as np

    # Input vector
    r = np.linspace(0, 100, 1024)
    f = np.zeros_like(r)
    f[1:] = scipy.special.jv(1, r[1:]) / r[1:]
    f[r == 0] = 0.5
    f = torch.tensor(f)

    fout = radial_2d_transform(f)

    rint = np.linspace(np.min(r), np.max(r), 100)
    fint = tf_generalSpline_regular1DGrid(r, rint, f)
    plt.figure()
    plt.plot(r, f, "r-")
    plt.plot(rint, fint, "b*")
    plt.xlabel("Radius /m")
    plt.show()
