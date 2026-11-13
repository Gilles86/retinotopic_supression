"""Side analysis (pRF atlas website + negative pRFs): group maps on fsaverage and per-region summaries.

Reads ``derivatives/side_prf_atlas_signed/sub-XX/sub-XX_space-fsaverage_maps.npz`` (``fit_signed_prf.py``)
for sub-03..sub-30 and writes

- ``derivatives/side_prf_atlas_group/group_maps_fsaverage.npz``: GROUP-level maps only, each ``(2, 163842)``
  (L, R); this is what the website is built from (no single-subject maps leave the cluster).
- ``derivatives/side_prf_atlas_group/region_stats.tsv``: one row per (subject, region, hemisphere) with
  per-subject summary numbers (fractions, medians), for the negative-pRF question.
- ``derivatives/side_prf_atlas_group/group_info.json``.

Reliability masks (no R² mixture / p_signal anywhere):

- Model-free: a subject's vertex is reliable when its cross-run reliability beats the circular-shift null,
  ``rel_p < .05`` and ``rel_r > 0``. Group mask ``rel_mask``: the number of reliable subjects exceeds
  chance (binomial vs 0.05, one-sided), Benjamini-Hochberg q < .01 over cortex.
- Cross-validated signed pRF: ``cv_p < .05`` and ``cv_r > 0`` (held-out session, against 50 shifted-stimulus
  fits). Group mask ``cv_mask``: same binomial test, q < .01.
- Negative / positive pRFs: a subject's vertex counts when it is cross-validated AND the amplitude
  is negative (positive) in the all-runs fit and in both session fits. Group masks: binomial vs 0.025, q < .01.

Group parameter maps (model 4: DoG with flexible HRF) use, per vertex, only the subjects reliable at that
vertex (model-free mask) and need at least 5 of them. Polar angle: circular mean of the subjects'
angles; shown where the Rayleigh test across those subjects gives p < .05. Eccentricity, sigma, HRF delay,
amplitude: medians.

Usage (compute node)::

    python -m retsupp.side_prf_atlas.group_fsaverage --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

SUBJECTS = range(3, 31)
OUT_SUBDIR = 'side_prf_atlas_group'
MIN_SUBJ = 5
SD_MAX = 6.0


def bh(p, q):
    """Benjamini-Hochberg over finite p; returns a boolean mask."""
    out = np.zeros(p.shape, bool)
    ok = np.isfinite(p)
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    passed = ps[order] <= q * np.arange(1, m + 1) / m
    if passed.any():
        thr = ps[order][np.nonzero(passed)[0].max()]
        out[ok] = ps <= thr
    return out


def binom_mask(k, n, p0, q=.01):
    p = np.where(n >= 10, stats.binom.sf(k - 1, np.maximum(n, 1), p0), np.nan)
    return bh(p, q), p


def circ_stats(x, y, use):
    """Circular mean angle, resultant length and Rayleigh p over subjects (axis 0)."""
    th = np.arctan2(y, x)
    c = np.where(use, np.cos(th), 0).sum(0)
    s = np.where(use, np.sin(th), 0).sum(0)
    n = use.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        R = np.sqrt(c ** 2 + s ** 2) / n
        Z = n * R ** 2
        p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))   # Zar's Rayleigh p
    return np.arctan2(s, c), R, np.where(n >= MIN_SUBJ, p, np.nan), n


def masked_median(v, use):
    w = np.where(use, v, np.nan)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(w, 0)
    return np.where(use.sum(0) >= MIN_SUBJ, med, np.nan)


def regions(fs_root):
    """Vertex masks on fsaverage: Yeo-7 networks crossed with Desikan-Killiany (aparc) regions."""
    out = {}
    for h, fh in enumerate(('lh', 'rh')):
        yl, _, yn = nib.freesurfer.read_annot(str(fs_root / 'fsaverage' / 'label' / f'{fh}.Yeo2011_7Networks_N1000.annot'))
        al, _, an = nib.freesurfer.read_annot(str(fs_root / 'fsaverage' / 'label' / f'{fh}.aparc.annot'))
        yn = [n.decode() for n in yn]
        an = [n.decode() for n in an]
        yeo = np.array([yn[i] if i >= 0 else '' for i in yl])
        ap = np.array([an[i] if i >= 0 else '' for i in al])
        dmn = yeo == '7Networks_7'
        defs = {
            'DMN (all)': dmn,
            'DMN medial parietal (PCC, isthmus, precuneus)': dmn & np.isin(ap, ['posteriorcingulate', 'isthmuscingulate', 'precuneus']),
            'DMN angular / inferior parietal': dmn & np.isin(ap, ['inferiorparietal', 'supramarginal']),
            'DMN lateral temporal': dmn & np.isin(ap, ['middletemporal', 'superiortemporal', 'bankssts', 'inferiortemporal']),
            'DMN medial prefrontal': dmn & np.isin(ap, ['superiorfrontal', 'medialorbitofrontal', 'rostralanteriorcingulate',
                                                       'caudalanteriorcingulate', 'frontalpole']),
            'Visual network (Yeo 1)': yeo == '7Networks_1',
            'Early visual (pericalcarine, cuneus, lingual; Yeo 1)': (yeo == '7Networks_1') & np.isin(ap, ['pericalcarine', 'cuneus', 'lingual']),
            'Dorsal attention (Yeo 3)': yeo == '7Networks_3',
        }
        for k, m in defs.items():
            out[(k, 'LR'[h])] = m
        out[('_yeo', 'LR'[h])] = yeo
    return out


def main(bids_folder):
    bids_folder = Path(bids_folder)
    der = bids_folder / 'derivatives'
    out_dir = der / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_out = out_dir / 'group_maps_fsaverage.npz'
    if maps_out.exists():
        print(f'{maps_out} exists, nothing to do')
        return
    subs, data = [], {}
    for s in SUBJECTS:
        f = der / 'side_prf_atlas_signed' / f'sub-{s:02d}' / f'sub-{s:02d}_space-fsaverage_maps.npz'
        if not f.exists():
            print(f'missing sub-{s:02d}')
            continue
        d = np.load(f)
        subs.append(s)
        for k in d.files:
            data.setdefault(k, []).append(d[k])
    D = {k: np.stack(v) for k, v in data.items()}             # (S, 2, 163842)
    S = len(subs)
    print(f'{S} subjects: {subs}')

    valid = np.isfinite(D['cv_r']) & np.isfinite(D['rel_r'])
    n = valid.sum(0)
    rel = valid & (D['rel_p'] < .05) & (D['rel_r'] > 0)
    cvs = valid & (D['cv_p'] < .05) & (D['cv_r'] > 0)
    neg = cvs & (D['amplitude'] < 0) & (D['sign_f1'] < 0) & (D['sign_f2'] < 0)
    pos = cvs & (D['amplitude'] > 0) & (D['sign_f1'] > 0) & (D['sign_f2'] > 0)

    G = {'n_subjects': n.astype(np.float32)}
    G['rel_prev'] = rel.sum(0) / np.maximum(n, 1)
    G['rel_mask'], G['rel_binom_p'] = binom_mask(rel.sum(0), n, .05)
    G['cv_prev'] = cvs.sum(0) / np.maximum(n, 1)
    G['cv_mask'], G['cv_binom_p'] = binom_mask(cvs.sum(0), n, .05)
    G['neg_prev'] = neg.sum(0) / np.maximum(n, 1)
    G['neg_mask'], G['neg_binom_p'] = binom_mask(neg.sum(0), n, .025)
    G['pos_prev'] = pos.sum(0) / np.maximum(n, 1)
    G['pos_mask'], G['pos_binom_p'] = binom_mask(pos.sum(0), n, .025)
    with np.errstate(invalid='ignore', divide='ignore'):
        G['sign_balance'] = (pos.sum(0) - neg.sum(0)) / (pos.sum(0) + neg.sum(0))
    G['median_rel_r'] = np.nanmedian(np.where(valid, D['rel_r'], np.nan), 0)
    G['median_cv_r'] = np.nanmedian(np.where(valid, D['cv_r'], np.nan), 0)
    G['median_m4_r2'] = np.nanmedian(D['m4_r2'], 0)

    use = rel & np.isfinite(D['m4_x'])
    ang, R, rp, nu = circ_stats(D['m4_x'], D['m4_y'], use)
    G['m4_polar_angle'], G['m4_angle_R'], G['m4_angle_rayleigh_p'], G['m4_n_used'] = ang, R, rp, nu.astype(np.float32)
    G['m4_ecc'] = masked_median(np.hypot(D['m4_x'], D['m4_y']), use)
    for p in ('sd', 'hrf_delay', 'hrf_dispersion', 'amplitude', 'srf_amplitude', 'srf_size'):
        G[f'm4_{p}'] = masked_median(D[f'm4_{p}'], use)
    for lab, sel in (('neg', neg), ('pos', pos)):
        a, Rr, pr, nn = circ_stats(D['x'], D['y'], sel)
        G[f'{lab}_polar_angle'], G[f'{lab}_angle_R'], G[f'{lab}_angle_rayleigh_p'] = a, Rr, pr
        G[f'{lab}_ecc'] = masked_median(np.hypot(D['x'], D['y']), sel)
        G[f'{lab}_sd'] = masked_median(D['sd'], sel)
    np.savez_compressed(maps_out, **{k: np.asarray(v, np.float32) for k, v in G.items()}, subjects=np.array(subs))

    # Coordinate-convention check: signed-fit vs model-4 positions where both are positive and reliable.
    chk = pos & rel & np.isfinite(D['m4_x'])
    print('signed-fit vs model-4 position, r(x) = %.3f, r(y) = %.3f (n = %d subject-vertices)' % (
        np.corrcoef(D['x'][chk], D['m4_x'][chk])[0, 1], np.corrcoef(D['y'][chk], D['m4_y'][chk])[0, 1], chk.sum()))

    fs_root = der / 'fmriprep' / 'sourcedata' / 'freesurfer'
    regs = regions(fs_root)
    rows = []
    for (name, hemi), m in regs.items():
        if name.startswith('_'):
            continue
        h = 'LR'.index(hemi)
        for i, s in enumerate(subs):
            v = valid[i, h] & m
            nv = v.sum()
            row = dict(subject=s, region=name, hemi=hemi, n_vertices=int(nv),
                       frac_rel=rel[i, h][m].sum() / max(nv, 1), frac_cv=cvs[i, h][m].sum() / max(nv, 1),
                       frac_neg=neg[i, h][m].sum() / max(nv, 1), frac_pos=pos[i, h][m].sum() / max(nv, 1))
            for lab, sel in (('neg', neg[i, h] & m), ('pos', pos[i, h] & m)):
                k = sel.sum()
                row[f'n_{lab}'] = int(k)
                if k == 0:
                    continue
                x, y, sd = D['x'][i, h][sel], D['y'][i, h][sel], D['sd'][i, h][sel]
                contra = (x > 0) if hemi == 'L' else (x < 0)
                ipsi = (x < 0) if hemi == 'L' else (x > 0)
                row[f'{lab}_laterality'] = (ipsi.sum() - contra.sum()) / max(ipsi.sum() + contra.sum(), 1)
                row[f'{lab}_frac_contra'] = contra.mean()
                row[f'{lab}_median_ecc'] = np.median(np.hypot(x, y))
                row[f'{lab}_median_sd'] = np.median(sd)
                row[f'{lab}_frac_sd_max'] = np.mean(sd >= SD_MAX * .999)
                row[f'{lab}_frac_beats_global'] = np.mean(D['cv_r'][i, h][sel] > D['glob_cv_r'][i, h][sel])
                row[f'{lab}_median_cv_r'] = np.median(D['cv_r'][i, h][sel])
                row[f'{lab}_median_glob_cv_r'] = np.median(D['glob_cv_r'][i, h][sel])
                d_ang = (np.arctan2(D['y_f1'][i, h][sel], D['x_f1'][i, h][sel])
                         - np.arctan2(D['y_f2'][i, h][sel], D['x_f2'][i, h][sel]))
                row[f'{lab}_session_angle_cos'] = np.mean(np.cos(d_ang))
                row[f'{lab}_session_x_sign_agree'] = np.mean(np.sign(D['x_f1'][i, h][sel]) == np.sign(D['x_f2'][i, h][sel]))
                if k >= 10:
                    row[f'{lab}_ecc_sd_spearman'] = stats.spearmanr(np.hypot(x, y), sd)[0]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'region_stats.tsv', sep='\t', index=False)
    info = dict(subjects=subs, n_subjects=S, min_subjects=MIN_SUBJ,
                masks=dict(rel='rel_p<.05 & rel_r>0; binomial vs .05, BH q<.01',
                           cv='cv_p<.05 & cv_r>0; binomial vs .05, BH q<.01',
                           neg='cv mask & amplitude<0 in all-runs and both session fits; binomial vs .025, BH q<.01'),
                n_vertices={k: int(np.asarray(v).sum()) for k, v in G.items() if k.endswith('_mask')})
    (out_dir / 'group_info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info['n_vertices']))
    print(f'wrote {maps_out} and region_stats.tsv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    main(ap.parse_args().bids_folder)
