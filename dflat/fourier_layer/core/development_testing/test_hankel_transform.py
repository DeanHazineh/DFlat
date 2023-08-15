import numpy as np
from dflat.fourier_layer.core.ops_hankel import qdht, iqdht

if __name__ == "__main__":
    # Simple test demonstration about the accuracy of our implementation
    # The code matches the pyhank official release
    import matplotlib.pyplot as plt
    from pyhank import qdht as pyhank_qdht
    import scipy.special

    # Input vector
    r = np.linspace(0, 200, 1024)
    f = np.zeros_like(r)
    f[1:] = scipy.special.jv(1, r[1:]) / r[1:]
    f[r == 0] = 0.5

    # pyhank transform
    pykr, pyht = pyhank_qdht(r, f)
    print(pykr.shape, pyht.shape)

    # my hankel
    kr, ht = qdht(r, f[None])
    print(kr.shape, ht.shape)

    r2, f2 = iqdht(kr, ht)

    plt.figure()
    plt.plot(pykr, pyht, "r*")
    plt.plot(kr, ht[0], "k-")
    plt.xlim([0, 5])
    plt.xlabel("Radial wavevector /m$^{-1}$")

    plt.figure()
    plt.plot(r, f, "r*")
    plt.plot(r2, f2[0], "k-")
    plt.xlabel("Radius /m")

    plt.show()
