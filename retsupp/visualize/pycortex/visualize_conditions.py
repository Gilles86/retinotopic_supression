import argparse
import cortex
import numpy as np
import matplotlib.pyplot as plt
from neural_priors.utils.data import Subject, get_all_subject_ids
from utils import get_alpha_vertex
from tqdm.contrib.itertools import product
from itertools import product as product_
import pandas as pd
from pathlib import Path
from nilearn import surface
from retsupp.utils.data import Subject


def main(subject: int = 24, model: int = 4):

    r2_thr = 0.05

    pc_subject = f'retsupp.sub-{subject:02d}'

    bids_folder = Path('/data/ds-retsupp')
    sub = Subject(subject, bids_folder=bids_folder)

    pars = sub.get_prf_parameters_surface(model=model)


    mask = (pars['r2'] > r2_thr).values

    # r2_vertex = cortex.Vertex(pars['r2'].values, subject=pc_subject, hemi='lh', cmap='hot')
    r2_vertex_thr = get_alpha_vertex(pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=0.5, subject=f'retsupp.sub-{subject:02d}')

    pars['theta'] = np.arctan2(pars['y'], pars['x'])

    theta_r = np.mod(pars['theta'].loc['R'], 2 * np.pi)
    theta_r = np.clip((theta_r - (.25 * np.pi)) / (1.5*np.pi), 0, 1)

    theta_l = np.mod(pars['theta'].loc['L'] + np.pi, 2 * np.pi)
    # theta_l = np.mod(pars['theta'] + np.pi, 2 * np.pi)
    theta_l = -np.clip((theta_l - (.25 * np.pi)) / (1.5*np.pi), 0, 1) + 1


    theta_ = get_alpha_vertex(np.concatenate([theta_l, theta_r]), mask, cmap='hsv', vmin=0.0, vmax=1.0, subject=pc_subject)


    pars['ecc'] = np.sqrt(pars['x']**2 + pars['y']**2)
    theta = get_alpha_vertex(pars['theta'].values, mask, cmap='hsv', vmin=-np.pi, vmax=np.pi, subject=pc_subject)
    ecc = get_alpha_vertex(pars['ecc'].values, mask, cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject)
    sd = get_alpha_vertex(pars['sd'].values, mask, cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject)

    x = get_alpha_vertex(pars['x'].values, mask, cmap='coolwarm', vmin=-4.0, vmax=4.0, subject=pc_subject)
    y = get_alpha_vertex(pars['y'].values, mask, cmap='coolwarm', vmin=-4.0, vmax=4.0, subject=pc_subject)

    ds = {'theta': theta, 'r2_thr': r2_vertex_thr,
        'theta_': theta_, 'ecc': ecc,
        'x': x, 'y': y, 'sd': sd}

    if model == 6:
        vmax = np.quantile(pars['neural_baseline'].values[mask], 0.95)
        neural_baseline = get_alpha_vertex(pars['neural_baseline'].values, mask, cmap='viridis', vmin=0.0, vmax=vmax, subject=pc_subject)
        ds.update({'neural_baseline': neural_baseline})

        vmax = np.quantile(pars['surround_baseline'].values[mask], 0.95)
        surround_baseline = get_alpha_vertex(pars['surround_baseline'].values, mask, cmap='viridis', vmin=0.0, vmax=vmax, subject=pc_subject)
        ds.update({'surround_baseline': surround_baseline})



    try:
        inferred_pars = sub.get_inferred_prf_pars_surf()
        ecc_inferred = get_alpha_vertex(inferred_pars['eccen'].values, mask, cmap='nipy_spectral', vmin=0.0, vmax=4.0, subject=pc_subject)
        theta_inferred = get_alpha_vertex(inferred_pars['angle'].values, mask, cmap='hsv', vmin=0, vmax=180, subject=pc_subject)
        roi = get_alpha_vertex(inferred_pars['varea'].values, alpha=~inferred_pars['varea'].isnull().values, subject=pc_subject, cmap='tab20')

        ds.update(**{'ecc_inferred':ecc_inferred, 'roi':roi,
                    'theta_inferred':theta_inferred})

    except Exception as e:
        print(f"Could not load inferred pars for subject {subject}: {e}")
        ecc_inferred = None
        theta_inferred = None
        roi = None



    ds = cortex.Dataset(**ds)

    cortex.webshow(ds)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=int, default=24, help='Subject ID to visualize')
    argparser.add_argument('--model', type=int, default=4, help='Model')
    args = argparser.parse_args()
    main(subject=args.subject, model=args.model)