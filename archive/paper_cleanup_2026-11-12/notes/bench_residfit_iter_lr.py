"""ResidualFitter A/B: vary max_iter + lr.

Tests whether fitting omega with more iterations and/or a lower
learning rate produces a meaningfully different decoding result.
StimulusFitter is held constant (empty init, L2=0.5, lr=0.05).
"""
from __future__ import annotations

import time
import sys

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject
from retsupp.decode.decoder import (
    PRF_PARS_BY_MODEL, _bc, load_prf_pars, build_paradigm_and_grid, make_prf_model,
)


def main(roi: str = 'V1', subject: int = 23, session: int = 1, run: int = 1):
    sub = Subject(subject, bids_folder='/data/ds-retsupp')
    _, _, ResidualFitter, StimulusFitter = _bc(4)

    prf, masker = load_prf_pars(sub, model=4)
    masker.fit()
    roi_img = sub.get_retinotopic_roi(roi=roi, bold_space=True)
    roi_flat = masker.transform(roi_img).flatten().astype(bool)
    sd = prf['sd']; r2v = prf['r2']
    ecc = np.sqrt(prf['x']**2 + prf['y']**2)
    pars_list = PRF_PARS_BY_MODEL[4]
    finite = np.all(np.stack([np.isfinite(prf[p]) for p in pars_list]), axis=0)
    keep = roi_flat & finite & (sd > 0.05) & (r2v > 0.1) & (r2v < 0.999) & (ecc <= 4.5)
    order = np.argsort(-r2v[keep])
    voxel_idx = np.where(keep)[0][order[:200]]
    print(f'{roi} ses-{session} run-{run}: {len(voxel_idx)} voxels '
          f'(sd>0.05, 0.1<r2<0.999, ecc<=4.5, top-200 by r2)')
    pars_df = pd.DataFrame({p: prf[p][voxel_idx] for p in pars_list})

    cache = np.load('/tmp/sub-23_kind-full_res-50.npz')
    bold = cache['bold'][:258][:, voxel_idx].astype(np.float32)
    paradigm = cache['paradigm'][:258].astype(np.float32)
    grid = cache['grid_coords']

    bold_df = pd.DataFrame(bold)
    paradigm_df = pd.DataFrame(paradigm)
    par_t = paradigm.reshape(258, 50, 50)

    def run_cell(resid_max_iter: int, resid_lr: float, label: str):
        m = make_prf_model(4, grid, paradigm, pars_df, data=bold_df)
        rf = ResidualFitter(model=m, data=bold_df, paradigm=paradigm_df,
                             parameters=pars_df.astype(np.float32))
        t0 = time.time()
        omega, _ = rf.fit(max_n_iterations=resid_max_iter,
                           learning_rate=resid_lr,
                           progressbar=False)
        omega = np.asarray(omega)
        t_o = time.time() - t0
        p = rf.fitted_omega_parameters
        print(f'  [{label}] omega fit: {t_o:5.1f}s  '
              f'rho={p["rho"]:.4f}  sigma2={p["sigma2"]:.4f}  '
              f'tau_med={np.median(p["tau"]):.3f}  '
              f'cond={np.linalg.cond(omega):.2e}')

        m2 = make_prf_model(4, grid, paradigm, pars_df, data=bold_df)
        sf = StimulusFitter(model=m2, data=bold_df, omega=omega,
                             parameters=pars_df.astype(np.float32))
        t0 = time.time()
        dec = sf.fit(l2_norm=0.5, learning_rate=0.05,
                      max_n_iterations=1000, min_n_iterations=200,
                      progressbar=False)
        t_d = time.time() - t0
        dec = dec.values.reshape(258, 50, 50)
        sig = dec[par_t > 0.5].mean()
        noi = dec[par_t <= 0.5].mean()
        corrs = []
        for t in range(258):
            if par_t[t].std() < 1e-9 or dec[t].std() < 1e-9:
                continue
            corrs.append(np.corrcoef(dec[t].ravel(), par_t[t].ravel())[0, 1])
        print(f'         StimulusFitter: {t_d:5.1f}s  max={dec.max():.3f}  '
              f'sig={sig:.4f}  noise={noi:.4f}  '
              f'snr={sig-noi:.4f}  corr={float(np.nanmean(corrs)):.4f}')
        return omega

    print('\n=== ResidualFitter sweep (decode with L2=0.5, lr=0.05, empty init) ===')
    o_A = run_cell(300, 0.02, 'A: 300  / 0.02 (current)')
    o_B = run_cell(2000, 0.02, 'B: 2000 / 0.02         ')
    o_C = run_cell(2000, 0.005, 'C: 2000 / 0.005        ')
    o_D = run_cell(4000, 0.002, 'D: 4000 / 0.002        ')

    print(f'\n||omega_B - omega_A||_fro / ||omega_A|| = '
          f'{np.linalg.norm(o_B - o_A) / np.linalg.norm(o_A):.5f}')
    print(f'||omega_C - omega_A||_fro / ||omega_A|| = '
          f'{np.linalg.norm(o_C - o_A) / np.linalg.norm(o_A):.5f}')
    print(f'||omega_D - omega_A||_fro / ||omega_A|| = '
          f'{np.linalg.norm(o_D - o_A) / np.linalg.norm(o_A):.5f}')


if __name__ == '__main__':
    roi = sys.argv[1] if len(sys.argv) > 1 else 'V1'
    main(roi=roi)
