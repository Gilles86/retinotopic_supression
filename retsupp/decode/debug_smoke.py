"""Quick interactive-ish probe of the smoke test pipeline.

Prints things like data scale, omega scale, predicted vs observed BOLD
correlation, whether the model can reproduce data given paradigm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from retsupp.utils.data import Subject
from retsupp.decode.decoder import (
    load_prf_pars, select_roi_voxels, load_run_bold,
    build_paradigm_and_grid, make_dog_model, _bc,
)


def main():
    sub = Subject(2, '/data/ds-retsupp')
    session, run = 1, 1
    resolution = 30
    max_voxels = 200

    prf, masker = load_prf_pars(sub, model=4)
    voxel_idx = select_roi_voxels(sub, roi='V1', prf=prf, masker=masker,
                                  sd_min=0.5, r2_min=0.05,
                                  max_voxels=max_voxels)
    print(f'Voxels kept: {len(voxel_idx)}')

    pars_df = pd.DataFrame({p: prf[p][voxel_idx] for p in
                            ['x', 'y', 'sd', 'amplitude', 'baseline',
                             'srf_amplitude', 'srf_size',
                             'hrf_delay', 'hrf_dispersion']})
    bold = load_run_bold(sub, session, run, masker)[:, voxel_idx]
    paradigm, grid = build_paradigm_and_grid(sub, session, run, resolution)

    print(f'  BOLD scale: mean {bold.mean():.3f}, std {bold.std():.3f}, '
          f'min {bold.min():.3f}, max {bold.max():.3f}')
    print(f'  PRF amp scale: mean {pars_df["amplitude"].mean():.3f}, '
          f'std {pars_df["amplitude"].std():.3f}')
    print(f'  PRF baseline scale: mean {pars_df["baseline"].mean():.3f}, '
          f'std {pars_df["baseline"].std():.3f}')
    print(f'  PRF r2 (kept) median: {np.median(prf["r2"][voxel_idx]):.3f}, '
          f'  max: {prf["r2"][voxel_idx].max():.3f}')

    bold_df = pd.DataFrame(bold, index=pd.Index(np.arange(bold.shape[0]),
                                                name='frame'))
    paradigm_df = pd.DataFrame(paradigm,
                               index=pd.Index(np.arange(paradigm.shape[0]),
                                              name='frame'))

    model = make_dog_model(grid, paradigm, pars_df, data=bold_df)

    pred = model.predict(paradigm=paradigm_df, parameters=pars_df.astype(np.float32))
    print(f'\n  PRED scale: mean {pred.values.mean():.3f}, '
          f'std {pred.values.std():.3f}')
    print(f'  PRED-DATA correlation per voxel (mean): '
          f'{np.array([np.corrcoef(pred.values[:, v], bold[:, v])[0,1] for v in range(min(20, bold.shape[1]))]).mean():.3f}')

    residuals = bold - pred.values
    print(f'\n  RESID scale: mean {residuals.mean():.3f}, '
          f'std {residuals.std():.3f}')

    # Fit residuals
    from braincoder.optimize import ResidualFitter, StimulusFitter
    rf = ResidualFitter(model=model, data=bold_df, paradigm=paradigm_df,
                        parameters=pars_df.astype(np.float32))
    omega, _ = rf.fit(max_n_iterations=300, progressbar=True)
    print(f'\n  OMEGA shape: {omega.shape}')
    print(f'  OMEGA diag: mean {np.diag(omega).mean():.3f}, '
          f'min {np.diag(omega).min():.3f}, max {np.diag(omega).max():.3f}')

    # Now try StimulusFitter with diff settings
    print('\n=== Try StimulusFitter (lr=0.5, l2=0.001, max_iter=300, min_iter=300) ===')
    model2 = make_dog_model(grid, paradigm, pars_df, data=bold_df)
    sf = StimulusFitter(model=model2, data=bold_df, omega=omega,
                        parameters=pars_df.astype(np.float32))
    decoded = sf.fit(l2_norm=0.001, learning_rate=0.5,
                     max_n_iterations=300, min_n_iterations=300,
                     progressbar=True)
    print(f'  Decoded scale: mean {decoded.values.mean():.3e}, '
          f'std {decoded.values.std():.3e}, max {decoded.values.max():.3e}')

    # Compare decoded peak position to true bar position at a frame where bar is on:
    par = sub.get_stimulus(session=session, run=run,
                           resolution=resolution).astype(np.float32)
    par2d = par.reshape(par.shape[0], -1)
    on_frames = np.where(par2d.sum(axis=1) > 0)[0]
    print(f'\n  bar-on frames: {len(on_frames)}, e.g. {on_frames[:5]}')
    if len(on_frames) > 0:
        for t in on_frames[:5]:
            true_bar_idx = np.argmax(par2d[t])
            decoded_idx = np.argmax(decoded.values[t])
            true_pos = grid[true_bar_idx]
            decoded_pos = grid[decoded_idx]
            print(f'    TR {t}: true peak at {true_pos}, '
                  f'decoded peak at {decoded_pos}, '
                  f'decoded max {decoded.values[t].max():.3e}')


if __name__ == '__main__':
    main()
