"""Supplementary figure: the decoded bar intensity map of Fig. 6c, measured next to the joint gain-field fit's prediction.

Inputs: ``notes/data/decoding_maps_jointhrf_2026-11.npz`` (Fig. 6c, all decoder voxels) and the per-subject files of
``retsupp/decode/decode_model_prediction.py`` (rsynced to ``notes/data/decoded_model_pred_2026-11/``): the same decoder
on the voxels the joint fit has, applied to the real data (``measured``) and to the fit's noise-free prediction
(``model``). Each map is computed exactly as Fig. 6c (``fig6_decoding.rel_maps`` / ``panel_map``): per subject the
HP-rotated map minus its average over the four 90° rotations, in % of the group mean of that average, mean over V1–V3,
HP at the top, smoothed for display; one colour scale. Sub-01/02 are left out (n = 28).

    python -m retsupp.visualize.paper.figS_decoded_map_model_vs_measured
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from retsupp.visualize.paper.fig6_decoding import APERTURE, EARLY, panel_map, rel_maps
from retsupp.visualize.paper.paper_style import DOUBLE, FG, save

DATA = Path('notes/data')


def quadrant_means(rm):
    """Per subject: mean of the HP-up relative map in the quadrant facing HP, the two next to it and the opposite one."""
    c = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(c, c)
    inside = np.hypot(X, Y) <= APERTURE
    ang = np.degrees(np.arctan2(Y, X))                       # HP at +90°
    q = {'hp': np.abs(ang - 90) <= 45, 'opposite': np.abs(ang + 90) <= 45}
    q['next'] = ~q['hp'] & ~q['opposite']
    return {k: np.array([np.nanmean(m[inside & v]) for m in rm]) for k, v in q.items()}


def main(name, exclude=(1, 2), vmax=3.):
    fig6 = dict(np.load(DATA / 'decoding_maps_jointhrf_2026-11.npz'))
    keep = ~np.isin(fig6['subjects'], exclude)
    fig6['drive'] = fig6['drive'][keep]
    rm6 = rel_maps(fig6)
    files = sorted((DATA / 'decoded_model_pred_2026-11').glob('sub-*_maps.npz'))
    subs = [int(f.stem.split('_')[0][4:]) for f in files]
    files = [f for f, s in zip(files, subs) if s not in exclude]
    rms = {}
    for i, kind in enumerate(('measured', 'model')):
        drive = []
        for f in files:
            z = np.load(f)
            ri = [list(z['rois']).index(r) for r in EARLY]
            drive.append(np.nanmean(z['drive'][i][ri], axis=1))          # mean over runs: (roi, y, x)
        maps = dict(drive=np.array(drive)[:, :, None], decoders=np.array(['free']), rois=np.array(EARLY))
        rms[kind] = rel_maps(maps, 'free', EARLY)
    nvox = np.mean([np.load(f)['n_voxels'].sum() / max(np.load(f)['n_decoder_voxels'].sum(), 1) for f in files])

    W = DOUBLE
    pw, gap, left = 1.75, .35, .25
    H = pw + .75
    fig = plt.figure(figsize=(W, H))
    titles = [('Measured (Fig. 6c)', rm6), ('Measured, joint-fit voxels', rms['measured']),
              ('Joint gain-field fit,\nnoise-free prediction', rms['model'])]
    for k, (t, rm) in enumerate(titles):
        ax = fig.add_axes([(left + k * (pw + gap)) / W, .2 / H, pw / W, pw / H])
        im = panel_map(ax, rm, vmax=vmax)
        ax.set_title(f'{t}\n(n = {len(rm)})', fontsize=7, pad=2)
        ax.text(-.02, 1.08, 'abc'[k], transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    cax = fig.add_axes([(left + 3 * (pw + gap) - .1) / W, .45 / H, .09 / W, (pw - .5) / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Decoded bar intensity,\n% vs rotation average', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(.5)

    qs = {k: quadrant_means(rm) for k, rm in (('fig6', rm6), ('measured', rms['measured']), ('model', rms['model']))}
    lines = []
    for k, lab in (('fig6', 'a'), ('measured', 'b'), ('model', 'c')):
        q = qs[k]
        d1, d2 = q['hp'] - q['next'], q['next'] - q['opposite']
        lines.append(f'{lab}: HP quadrant {np.mean(q["hp"]):+.2f} ± {stats.sem(q["hp"]):.2f}%, next {np.mean(q["next"]):+.2f} ± '
                     f'{stats.sem(q["next"]):.2f}%, opposite {np.mean(q["opposite"]):+.2f} ± {stats.sem(q["opposite"]):.2f}%; '
                     f'HP − next {np.mean(d1):+.2f}% (Wilcoxon p = {stats.wilcoxon(d1).pvalue:.3g}), next − opposite '
                     f'{np.mean(d2):+.2f}% (p = {stats.wilcoxon(d2).pvalue:.3g})')
    gap_c = {k: qs['measured'][k] - qs['model'][k] for k in ('hp', 'next', 'opposite')}
    g1 = (qs['measured']['hp'] - qs['measured']['next']) - (qs['model']['hp'] - qs['model']['next'])
    g2 = (qs['measured']['next'] - qs['measured']['opposite']) - (qs['model']['next'] - qs['model']['opposite'])
    legend = f"""
**Figure S. The decoded bar map, measured and predicted by the joint gain-field fit.** a, The map of Fig. 6c: the bar decoded from V1–V3 with the gain-free decoder (pRFs only, no gain fields), per pixel the mean decoded intensity over the frames the bar covers it, each run rotated so the high-probability (HP) distractor location is at the top, minus the average over the four 90° rotations, in % of that average, mean over V1–V3 and participants (n = 28), smoothed for display. b, The same decoder on the voxels that are also in the joint gain-field fit (on average {100 * nvox:.0f}% of the decoder's voxels). c, The same decoder applied to the joint fit's noise-free prediction of each run (the fitted model with all its gain fields and the eight search items, each voxel at the real run's mean, no noise), on the voxels of b, with the noise model of the real run. Same colour scale in all panels; red dashed circle, HP; grey circles, the other three locations. Quadrant means (mean ± SEM over participants; the quadrant facing HP, the two next to it, the opposite one): {'; '.join(lines)}. Measured (b) − predicted (c), paired over participants: HP − next {np.mean(g1):+.2f}% (Wilcoxon p = {stats.wilcoxon(g1).pvalue:.3g}); next − opposite {np.mean(g2):+.2f}% (p = {stats.wilcoxon(g2).pvalue:.3g}). The fit reproduces the lower intensity in the quadrant facing HP but not the higher intensity next to HP than opposite it.
"""
    import re
    legend = re.sub(r'(?<![\w])-(?=\d)', '−', legend)
    save(fig, name, legend)
    print(legend)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', default='figS_decoded_map_model_vs_measured_n28')
    a = ap.parse_args()
    main(a.name)
