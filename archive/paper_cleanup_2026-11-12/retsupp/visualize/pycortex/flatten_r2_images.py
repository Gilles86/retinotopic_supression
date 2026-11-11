
from pathlib import Path
from retsupp.utils.data import get_subject_ids, Subject
import numpy as np
from retsupp.visualize.utils import get_alpha_vertex
import cortex
import matplotlib.pyplot as plt

# Use BIDS folder and output to derivatives/prf_figures
bids_folder = Path('/data/ds-retsupp')
output_dir = bids_folder / 'derivatives' / 'prf_figures'
output_dir.mkdir(parents=True, exist_ok=True)

r2_thr = 0.05
model = 4
subjects = [int(e) for e in get_subject_ids()]

for subject in subjects:
    try:
        sub = Subject(subject)
        pc_subject = f'retsupp.sub-{subject:02d}'
        pars = sub.get_prf_parameters_surface(model=model, space='fsaverage')
        mask = (pars['r2'] > r2_thr).values
        r2_vertex_thr = get_alpha_vertex(pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=1.0, subject='fsaverage')
        # Flatten and plot using pycortex quickflat
        out_png = output_dir / f'sub-{subject:02d}_r2.png'
        cortex.quickflat.make_png(str(out_png), r2_vertex_thr, with_colorbar=False,
                                with_rois=False, with_labels=False)
        print(f'Saved r2 figure for subject {subject} to {out_png}')

        pars['theta'] = np.arctan2(pars['y'], pars['x'])

        theta_r = np.mod(pars['theta'].loc['R'], 2 * np.pi)
        theta_r = np.clip((theta_r - (.25 * np.pi)) / (1.5*np.pi), 0, 1)

        theta_l = np.mod(pars['theta'].loc['L'] + np.pi, 2 * np.pi)
        theta_l = -np.clip((theta_l - (.25 * np.pi)) / (1.5*np.pi), 0, 1) + 1
        theta_ = get_alpha_vertex(np.concatenate([theta_l, theta_r]), mask, cmap='hsv', vmin=0.0, vmax=1.0, subject='fsaverage')
        cortex.quickflat.make_png(str(output_dir / f'sub-{subject:02d}_theta.png'), theta_,
                                with_colorbar=False, with_rois=False, with_labels=False)
        print(f'Saved theta figure for subject {subject} to {out_png}')

    except Exception as e:
        print(f'Error processing subject {subject}: {e}')
