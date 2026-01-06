from retsupp.utils.data import get_subject_ids, Subject
import numpy as np
from utils import get_alpha_vertex
import cortex

subject_id = '01'

subjects = [int(e) for e in get_subject_ids()]
r2_thr = 0.1
model = 4


results = {}


for subject in subjects:
    sub = Subject(subject)
    pc_subject = f'retsupp.sub-{subject:02d}'

    pars = sub.get_prf_parameters_surface(model=model, space='fsaverage')

    mask = (pars['r2'] > r2_thr).values

    # r2_vertex = cortex.Vertex(pars['r2'].values, subject=pc_subject, hemi='lh', cmap='hot')
    r2_vertex_thr = get_alpha_vertex(pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=0.5, subject=f'fsaverage')

    pars['theta'] = np.arctan2(pars['y'], pars['x'])

    theta_r = np.mod(pars['theta'].loc['R'], 2 * np.pi)
    theta_r = np.clip((theta_r - (.25 * np.pi)) / (1.5*np.pi), 0, 1)

    theta_l = np.mod(pars['theta'].loc['L'] + np.pi, 2 * np.pi)
    theta_l = -np.clip((theta_l - (.25 * np.pi)) / (1.5*np.pi), 0, 1) + 1

    theta_ = get_alpha_vertex(np.concatenate([theta_l, theta_r]), mask, cmap='hsv', vmin=0.0, vmax=1.0, subject='fsaverage')

    # results[f'{subject:02d}.theta'] = theta_
    results[f'{subject}.r2'] = r2_vertex_thr

print(results)
ds = cortex.Dataset(**results)
cortex.webshow(ds)
