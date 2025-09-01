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


subject = 5
model = 1
r2_thr = {1:0.025, 3:0.08, 5:0.05}[subject]

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

ds = cortex.Dataset(**{'theta_':theta_, 'r2_thr':r2_vertex_thr,
                       'theta':theta, 'ecc':ecc,
                       'x':x, 'y':y, 'sd':sd})

cortex.webshow(ds)