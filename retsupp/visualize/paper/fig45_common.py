"""Shared loading and statistics for Figures 4 and 5 and Table 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from retsupp.visualize.paper.paper_style import ROI_ORDER


def joint(table, family, model='m7_full'):
    d = pd.read_csv(table, sep='\t')
    d = d[(d.family == family) & (d.model == model) & d.roi.isin(ROI_ORDER)].copy()
    d['dS'] = d.g_HP - d.g_LP
    d['dD'] = d.g_HP_dyn - d.g_LP_dyn
    d['T'] = d.g_T_dyn
    d['D'] = (d.g_HP_dyn + 3 * d.g_LP_dyn) / 4   # distractor gain averaged over locations, weighted by location count
    return d


def conditional(table):
    d = pd.read_csv(table, sep='\t')
    d = d[d.valid & d.roi.isin(ROI_ORDER)]
    return d.pivot_table(index=['subject', 'roi'], columns='mech', values='estimate').reset_index()


def median_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x)[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), *np.percentile(boots, [2.5, 97.5])


def jzs_bf10(x, r=np.sqrt(2) / 2):
    """One-sample JZS Bayes factor (Rouder et al. 2009) for mean != 0."""
    x = np.asarray(x)[np.isfinite(x)]
    n = len(x)
    t = stats.ttest_1samp(x, 0).statistic
    nu = n - 1
    num = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    f = lambda g: (1 + n * g * r ** 2) ** -.5 * (1 + t ** 2 / ((1 + n * g * r ** 2) * nu)) ** (-(nu + 1) / 2) * \
        (2 * np.pi) ** -.5 * g ** -1.5 * np.exp(-1 / (2 * g))
    den, _ = integrate.quad(f, 0, np.inf)
    return den / num
