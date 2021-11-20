import numpy as np
from collections import namedtuple
from scipy.stats import norm, vonmises

Phase = namedtuple('Phase', ['start', 'end', 'std', 'coeff'])

def vonmises_func(x, behavior, shift=0, xlim=1.0):

    out = 0
    x   = (x + shift) * 2 * np.pi
    for phase in behavior:
        x_0 = phase.start * 2 * np.pi
        x_1 = phase.end   * 2 * np.pi
        std = phase.std
        c   = phase.coeff

        kappa = 1 / (std ** 2)

        p1 = vonmises.cdf(x, kappa=kappa, loc=x_0, scale=xlim)
        p2 = vonmises.cdf(x, kappa=kappa, loc=x_1, scale=xlim)
        out += c * (p1 - p2)
    return out
