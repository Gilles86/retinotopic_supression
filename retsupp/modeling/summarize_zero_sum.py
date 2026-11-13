"""One row per (subject, area, zero-sum model, start) of a zero-sum LM root, next to the reference fits.

Reads ``<zs root>/{m4_TDS_zs,m7_full_zs}/sub-XX/*_cv-fits.pkl`` (``fit_cv_ladder_nested`` ZERO_SUM_MODELS): per start
(cold / warm / parent) the training SSE before and after LM, the shared terms, which start was kept and which terms
ended on a bound; and, for comparison, the same model's shared terms in ``--ref-root`` (m4_TDS / m7_full, e.g. the
refitted-polished reference). dS = g_HP − g_LP.

    python -m retsupp.modeling.summarize_zero_sum \\
        --zs-root af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_zs_lm_canary \\
        --ref-root af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished --out <tsv>
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import pandas as pd

TERMS = ['g_HP', 'g_LP', 'g_HP_dyn', 'g_LP_dyn', 'g_T_dyn', 'sigma_AF', 'sigma_dyn', 'item_scale']
REF_OF = {'m4_TDS_zs': 'm4_TDS', 'm7_full_zs': 'm7_full'}


def main(bids, zs_root, ref_root, out):
    deriv = Path(bids) / 'derivatives'
    rows = []
    for model, ref_model in REF_OF.items():
        for f in sorted(glob.glob(str(deriv / zs_root / model / 'sub-*' / '*_cv-fits.pkl'))):
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            info = d['optimizer_info'][0]
            sub = int(Path(f).parent.name.split('-')[1])
            roi = Path(f).name.split('_roi-')[1].split('_')[0]
            ref = {}
            rf = deriv / ref_root / ref_model / f'sub-{sub:02d}' / Path(f).name
            if rf.exists():
                with open(rf, 'rb') as fh:
                    rp = pickle.load(fh)['fit_pars_per_fold'][0]
                ref = {f'ref_{t}': float(rp[t].iloc[0]) for t in TERMS}
                ref['ref_dS'] = ref['ref_g_HP'] - ref['ref_g_LP']
                ref['ref_sse'] = (rp.attrs or {}).get('sse', float('nan'))
            for c in info['candidates']:
                row = dict(subject=sub, roi=roi, model=model, n_vox=len(d['voxel_ids']), start=c['start'],
                           kept=c['start'] == info['start'], sse_start=c['sse_start'], sse_final=c['sse_final'],
                           nit=c['nit'], seconds=c['seconds'], reason=c['reason'],
                           on_bound=','.join(info.get('on_bound', [])) if c['start'] == info['start'] else '',
                           **{t: c[t] for t in TERMS})
                row['dS'] = row['g_HP'] - row['g_LP']
                row.update(ref)
                rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out, sep='\t', index=False)
    kept = t[t.kept]
    print(kept[['subject', 'roi', 'model', 'n_vox', 'start', 'sse_final', 'dS', 'ref_dS', 'g_T_dyn', 'ref_g_T_dyn',
                'g_HP_dyn', 'g_LP_dyn', 'item_scale', 'ref_item_scale', 'sigma_AF', 'on_bound']
               ].round(4).to_string(index=False))
    both = t.groupby(['subject', 'roi', 'model']).sse_final.agg(['min', 'max'])
    print('\nSSE gap between the starts (max - min):')
    print((both['max'] - both['min']).describe().round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-retsupp')
    ap.add_argument('--zs-root', required=True)
    ap.add_argument('--ref-root', default='af_prf_cv_ladder_nested_full_bounded_maxsd20_reliable_roihrf_polished')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.zs_root, a.ref_root, a.out)
