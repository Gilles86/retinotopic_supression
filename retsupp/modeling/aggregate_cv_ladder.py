"""Aggregate the nested gain-field model comparison (``af_prf_cv_ladder``).

Reads every model arm under ``derivatives/af_prf_cv_ladder/<model>/`` (the
per-(subject, ROI) ``*_cv-r2.tsv`` written by ``fit_af_prf_cv_v2``), checks
that all arms scored identical voxels, adds the model-free null level
(mean training-run response; this LOADS BOLD, so run it cluster-side), and
writes three TSVs into ``derivatives/af_prf_cv_ladder/``:

* ``cv_ladder_per_voxel.tsv``  — subject, roi, voxel_id, cv_r2_null, cv_r2_<model>...
* ``cv_ladder_subject.tsv``    — per (subject, roi, model): mean CV-R² over signal
  voxels (null CV-R² > 0) and over all voxels.
* ``cv_ladder_stats.tsv``      — per (contrast, roi): mean/SEM/d_z of the
  subject-level ΔCV-R², Wilcoxon signed-rank p (two-sided), n subjects
  favouring the richer model, BH-FDR over ROIs within each contrast.

Usage::

    srun --account=zne.uzh --mem=64G --time=02:00:00 -c4 \\
        $HOME/data/conda/envs/retsupp_cuda/bin/python -m \\
        retsupp.modeling.aggregate_cv_ladder \\
        --bids-folder /shares/zne.uzh/gdehol/ds-retsupp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from retsupp.modeling.cv_helpers import per_voxel_r2
from retsupp.modeling.fit_af_prf_cv_v2 import (
    build_data_and_paradigm,
    make_folds,
    split_by_condition_with_target,
)
from retsupp.utils.data import Subject

ROI_ORDER = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO',
             'IPS', 'SPL1', 'FEF']


def _arm_per_voxel_cv_r2(tsv: Path) -> pd.DataFrame:
    """Read one arm's cv-r2.tsv -> per-voxel nanmean-over-folds cv_r2.

    Returns a DataFrame indexed by voxel_id with a single ``cv_r2`` column
    (mean over the up-to-4 folds, ignoring NaN/skipped folds).
    """
    df = pd.read_csv(tsv, sep='\t')
    agg = (df.groupby('voxel_id')['cv_r2']
             .apply(lambda s: np.nanmean(s.to_numpy(dtype=np.float64)))
             .rename('cv_r2'))
    return agg.to_frame()


def compute_null_cv_r2(
    bold_sel: np.ndarray, paradigm, condition_indicator,
    dynamic_indicator, target_indicator, run_meta, cv_scheme: str = 'loco',
) -> np.ndarray:
    """Per-voxel null CV-R² (mean over folds), MODEL-FREE.

    For each leave-one-condition-out fold the prediction for a held-out run
    is the **across-TRAINING-RUNS mean RESPONSE** — i.e. the BOLD averaged
    over the training runs at each *within-run* timepoint. This is the
    reproducible, condition-blind evoked response (the bar protocol is a
    fixed sequence across runs), NOT a scalar per-voxel mean. Scored with
    the same ``per_voxel_r2`` on the held-out BOLD.

    ``bold_sel`` is the subject's BOLD already sliced to the arms' voxel_ids
    (loaded once per subject by the caller). All other args are the
    subject-level paradigm / indicators / run_meta from
    ``build_data_and_paradigm``.
    """
    n_vox = bold_sel.shape[1]

    fold_r2 = []
    for _label, held_runs in make_folds(run_meta, cv_scheme):
        try:
            split = split_by_condition_with_target(
                bold=bold_sel, paradigm=paradigm,
                condition_indicator=condition_indicator,
                dynamic_indicator=dynamic_indicator,
                target_indicator=target_indicator,
                run_meta=run_meta, held_out_runs=held_runs)
        except RuntimeError:
            fold_r2.append(np.full(n_vox, np.nan, dtype=np.float64))
            continue

        train_bold = split['train']['bold']     # (n_tr*T_run, n_vox)
        held_bold = split['held']['bold']        # (n_he*T_run, n_vox)
        n_tr = len(split['train']['run_meta'])
        n_he = len(split['held']['run_meta'])
        # Reshape concatenated runs -> (n_runs, T_run, n_vox); needs equal
        # run length (258 within a subject). Guard and skip the fold if not.
        if (n_tr == 0 or n_he == 0
                or train_bold.shape[0] % n_tr or held_bold.shape[0] % n_he):
            fold_r2.append(np.full(n_vox, np.nan, dtype=np.float64))
            continue
        T_run = train_bold.shape[0] // n_tr
        if held_bold.shape[0] // n_he != T_run:
            fold_r2.append(np.full(n_vox, np.nan, dtype=np.float64))
            continue

        # Across-training-runs mean response (per within-run timepoint).
        mean_resp = train_bold.reshape(n_tr, T_run, n_vox).mean(axis=0)
        # Predict every held run with that same mean response.
        pred = np.tile(mean_resp, (n_he, 1))     # (n_he*T_run, n_vox)
        fold_r2.append(per_voxel_r2(held_bold, pred).astype(np.float64))

    stack = np.vstack(fold_r2)        # (n_folds, n_vox)
    with np.errstate(invalid='ignore'):
        return np.nanmean(stack, axis=0)

# (richer, simpler, label). Only contrasts whose two arms exist are tested.
CONTRASTS = [
    ('m0_prf', 'm00_bar', 'search items vs vanilla pRF'),
    ('m1_T', 'm0_prf', 'T vs pRF'),
    ('m2_D', 'm0_prf', 'D vs pRF'),
    ('m3_S', 'm0_prf', 'S vs pRF'),
    ('m3b_TD', 'm0_prf', 'T+D vs pRF'),
    ('m4_TDS', 'm3b_TD', 'S given T+D (three vs two components)'),
    ('m4_TDS', 'm0_prf', 'T+D+S vs pRF'),
    ('m5_TDS_dS', 'm4_TDS', 'sustained learning (dS)'),
    ('m6_TDS_dD', 'm4_TDS', 'dynamic learning (dD)'),
    ('m7_full', 'm4_TDS', 'any learning (dS+dD)'),
    ('m7_full', 'm5_TDS_dS', 'dD given dS'),
    ('m7_full', 'm6_TDS_dD', 'dS given dD'),
    ('m7_full', 'm7_full_minusT', 'T given rest'),
    ('m7_full', 'm7_full_minusD', 'D+dD given rest'),
    ('m7_full', 'm7_full_minusS', 'S+dS given rest'),
    ('m0_prf', 'm0_prf_itemsfixed', 'item scale (no field)'),
    ('m7_full', 'm7_full_itemsfixed', 'item scale (full)'),
]


def collect(root: Path, bids_folder: Path, with_null: bool,
            cv_scheme: str = 'loco',
            subjects: list[int] | None = None) -> pd.DataFrame:
    models = sorted(p.name for p in root.iterdir()
                    if p.is_dir() and not p.name.startswith('canary')
                    and any(p.glob('sub-*/*_cv-r2.tsv')))
    print(f'Models found: {models}')
    cells: dict[tuple[int, str], dict[str, Path]] = {}
    for m in models:
        for tsv in (root / m).glob('sub-*/sub-*_roi-*_cv-r2.tsv'):
            subject = int(tsv.name.split('_')[0].split('-')[1])
            roi = tsv.name.split('roi-')[1].split('_cv-r2')[0]
            cells.setdefault((subject, roi), {})[m] = tsv

    rows = []
    for subject in sorted({s for s, _ in cells}):
        if subjects is not None and subject not in subjects:
            continue
        bold = None
        if with_null:
            try:
                (bold, paradigm, ci, di, ti, _g, _m, run_meta) = \
                    build_data_and_paradigm(Subject(subject, bids_folder))
            except Exception as e:                       # noqa: BLE001
                print(f'  !! BOLD load failed sub-{subject:02d}: {e}')
        for roi in ROI_ORDER:
            arms = cells.get((subject, roi))
            if not arms:
                continue
            r2 = {m: _arm_per_voxel_cv_r2(p) for m, p in arms.items()}
            ref = next(iter(r2.values())).index.to_numpy()
            if any(not np.array_equal(v.index.to_numpy(), ref) for v in r2.values()):
                print(f'  !! voxel-id mismatch sub-{subject:02d} {roi}; skipping')
                continue
            block = pd.DataFrame({'subject': subject, 'roi': roi, 'voxel_id': ref})
            if bold is not None:
                block['cv_r2_null'] = compute_null_cv_r2(
                    bold[:, ref], paradigm, ci, di, ti, run_meta, cv_scheme)
            else:
                block['cv_r2_null'] = np.nan
            for m, v in r2.items():
                block[f'cv_r2_{m}'] = v['cv_r2'].to_numpy()
            rows.append(block)
            print(f'  sub-{subject:02d} {roi}: {len(ref)} vox, {len(r2)} models')
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def subject_level(vox: pd.DataFrame) -> pd.DataFrame:
    model_cols = [c for c in vox.columns
                  if c.startswith('cv_r2_') and c != 'cv_r2_null']
    out = []
    for (subject, roi), g in vox.groupby(['subject', 'roi']):
        sig = g['cv_r2_null'] > 0 if g['cv_r2_null'].notna().any() \
            else pd.Series(True, index=g.index)
        for c in model_cols:
            out.append(dict(subject=subject, roi=roi, model=c[len('cv_r2_'):],
                            n_vox=len(g), n_signal=int(sig.sum()),
                            cv_r2_signal=g.loc[sig, c].mean(),
                            cv_r2_signal_median=g.loc[sig, c].median(),
                            cv_r2_all=g[c].mean()))
    return pd.DataFrame(out)


def _holm(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def contrast_stats(subj: pd.DataFrame,
                   metric: str = 'cv_r2_signal_median') -> pd.DataFrame:
    """Per contrast x ROI: one-sample t-test on per-subject delta (as in
    retinonumeral fig. 2G), Wilcoxon as a check; Holm and BH across ROIs."""
    wide = subj.pivot_table(index=['subject', 'roi'], columns='model',
                            values=metric)
    out = []
    for rich, simple, label in CONTRASTS:
        if rich not in wide or simple not in wide:
            continue
        rows = []
        for roi in ROI_ORDER:
            if roi not in wide.index.get_level_values('roi'):
                continue
            d = (wide.xs(roi, level='roi')[rich]
                 - wide.xs(roi, level='roi')[simple]).dropna()
            if len(d) < 5:
                continue
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            t, p_t = stats.ttest_1samp(d, 0)
            rows.append(dict(contrast=label, richer=rich, simpler=simple,
                             roi=roi, n=len(d), mean_delta=d.mean(),
                             median_delta=d.median(),
                             sem=d.std(ddof=1) / np.sqrt(len(d)),
                             dz=d.mean() / d.std(ddof=1) if d.std() > 0 else np.nan,
                             n_richer_better=int((d > 0).sum()),
                             t=t, p_t=p_t, p_wilcoxon=p))
        if rows:
            r = pd.DataFrame(rows)
            r['p_t_holm'] = _holm(r['p_t'].to_numpy())
            r['p_t_fdr'] = _bh(r['p_t'].to_numpy())
            r['p_wilcoxon_fdr'] = _bh(r['p_wilcoxon'].to_numpy())
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--no-null', action='store_true',
                    help='Skip the BOLD-based null level (scores all voxels).')
    ap.add_argument('--cv-scheme', choices=['loco', 'run3'], default='loco')
    ap.add_argument('--subject', type=int, default=None,
                    help='Only collect this subject and write '
                         'per_subject/sub-XX_cv_ladder_per_voxel.tsv '
                         '(run as a SLURM array; then --merge).')
    ap.add_argument('--merge', action='store_true',
                    help='Concatenate per_subject/*.tsv instead of '
                         're-collecting, then write the three TSVs.')
    a = ap.parse_args()
    bids = Path(a.bids_folder)
    sub = 'af_prf_cv_ladder' + ('' if a.cv_scheme == 'loco' else f'_{a.cv_scheme}')
    root = bids / 'derivatives' / sub
    per_sub_dir = root / 'per_subject'
    if a.subject is not None:
        vox = collect(root, bids, with_null=not a.no_null,
                      cv_scheme=a.cv_scheme, subjects=[a.subject])
        if vox.empty:
            raise SystemExit(f'No complete cells for sub-{a.subject:02d}.')
        per_sub_dir.mkdir(exist_ok=True)
        out = per_sub_dir / f'sub-{a.subject:02d}_cv_ladder_per_voxel.tsv'
        vox.to_csv(out, sep='\t', index=False)
        print(f'Wrote {out}')
        return
    if a.merge:
        files = sorted(per_sub_dir.glob('sub-*_cv_ladder_per_voxel.tsv'))
        print(f'Merging {len(files)} per-subject TSVs')
        vox = pd.concat([pd.read_csv(f, sep='\t') for f in files],
                        ignore_index=True)
    else:
        vox = collect(root, bids, with_null=not a.no_null,
                      cv_scheme=a.cv_scheme)
    if vox.empty:
        raise SystemExit('No complete cells found.')
    vox.to_csv(root / 'cv_ladder_per_voxel.tsv', sep='\t', index=False)
    subj = subject_level(vox)
    subj.to_csv(root / 'cv_ladder_subject.tsv', sep='\t', index=False)
    st = contrast_stats(subj)
    st.to_csv(root / 'cv_ladder_stats.tsv', sep='\t', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 300):
        print(st.round(5).to_string(index=False))
    print(f'Wrote TSVs to {root}')


if __name__ == '__main__':
    main()
