"""CVD simulation (Machado et al. 2009, severity 1) and CIEDE2000 distances between palette roles."""
import itertools
import numpy as np
from matplotlib.colors import to_rgb
from skimage.color import rgb2lab, deltaE_ciede2000

M = {'Deuteranopia': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
     'Protanopia': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
     'Tritanopia': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]])}


def lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def sim_rgb(rgb, view):
    """rgb: (..., 3) in [0, 1]."""
    if view == 'Normal':
        return np.asarray(rgb, float)
    return srgb(lin(np.asarray(rgb, float)) @ M[view].T)


def sim(h, view):
    return tuple(sim_rgb(np.array(to_rgb(h)), view))


def de(a, b, view='Normal'):
    la = rgb2lab(np.array(sim(a, view))[None, None])
    lb = rgb2lab(np.array(sim(b, view))[None, None])
    return float(deltaE_ciede2000(la, lb)[0, 0])


VIEWS = ['Normal', 'Deuteranopia', 'Protanopia', 'Tritanopia']


def table(pairs, cols):
    rows = []
    for a, b in pairs:
        rows.append((a, b, *[de(cols[a], cols[b], v) for v in VIEWS]))
    return rows
