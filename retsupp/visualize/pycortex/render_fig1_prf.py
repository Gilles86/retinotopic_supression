"""Pycortex renders of one subject's pRF maps on the inflated surface, for figure 1.

Polar angle, eccentricity, pRF size and R² of the mean-across-runs m4 fit, with the
neuropythy visual-area borders. Colour and borders are baked per vertex with curvature
(``VertexRGB``). The subjects have no flatmap and so no ROI overlay.

The viewer's ``surface.<subject>.<control>`` paths split on dots, so ``retsupp.sub-23``
cannot be driven. The script links an alias without dots in the pycortex database and
removes the link afterwards; its cache files stay in the subject's ``cache/`` folder.

``cortex.export.save_3d_views`` needs a browser, so this opens the viewer in headless
Chrome and leaves the user's own browser alone. Runs in pycortex2::

    ~/mambaforge/envs/pycortex2/bin/python \\
        retsupp/visualize/pycortex/render_fig1_prf.py 23 --out-dir notes/figures/fig1_inflated
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

import cortex
import matplotlib as mpl
import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_morph_data
from scipy import sparse

BIDS = Path('/data/ds-retsupp')
FS = BIDS / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
R2_RAMP = (.05, .20)         # R² -> opacity of the colour over the curvature
ECC_MAX_SHOWN = 5.0          # stimulus grid radius
CURV = dict(brightness=.55, contrast=.17, smooth=20)
PANELS = {'angle': ('hsv', 0, 1), 'ecc': ('nipy_spectral', 0, 4),
          'sd': ('nipy_spectral', 0, 3), 'r2': ('inferno', 0, .6)}
BASE_VIEW = {'camera.target': [0, 0, 0], 'camera.radius': 190, 'surface.{subject}.unfold': 1,
             'surface.{subject}.specularity': 0}
VIEWS = {
    'back': {'camera.azimuth': 180, 'camera.altitude': 90,
             'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'medial': {'camera.azimuth': 0, 'camera.altitude': 90,
               'surface.{subject}.pivot': 180, 'surface.{subject}.shift': 10},
    'pole': {'camera.azimuth': 180, 'camera.altitude': 90, 'camera.radius': 190,
             'camera.target': [0, 0, -6],
             'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'dorsal': {'camera.azimuth': 180, 'camera.altitude': 35, 'camera.radius': 225,
               'camera.target': [0, -10, 0],
               'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'ventral': {'camera.azimuth': 180, 'camera.altitude': 145, 'camera.radius': 165,
                'camera.target': [0, -75, 0],
                'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'fef_left': {'camera.azimuth': 75, 'camera.altitude': 35, 'camera.radius': 150,
                 'camera.target': [-25, 25, 20],
                 'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'fef_right': {'camera.azimuth': 285, 'camera.altitude': 35, 'camera.radius': 150,
                  'camera.target': [25, 25, 20],
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'lateral': {'camera.azimuth': 180, 'camera.altitude': 90,
                'surface.{subject}.pivot': 180, 'surface.{subject}.shift': 10},
}
IMAGE_SIZE = (2400, 1800)
# Visual areas of the neuropythy atlas, grouped as in the analyses.
AREA_COLORS = {'V1': ((1,), '#3D5A80'), 'V2': ((2,), '#2A9D8F'), 'V3': ((3,), '#8AB17D'),
               'hV4': ((4,), '#E9C46A'), 'VO': ((5, 6), '#F4A261'), 'LO': ((7, 8), '#E76F51'),
               'TO': ((9, 10), '#B5179E'), 'V3AB': ((11, 12), '#7209B7'),
               'IPS': ((101,), '#0077B6'), 'FEF': ((102,), '#D62828')}
# Wang 2015 labels added to the neuropythy areas; the neuropythy label wins on overlap,
# as in the analyses. Output codes as in AREA_COLORS.
WANG = {101: (18, 19, 20, 21, 22, 23), 102: (25,)}


def theta_value(theta, hemi):
    """Visual-field angle -> hsv index, as ``visualize_prf._theta_color_value``."""
    if hemi == 'R':
        v = np.mod(theta, 2 * np.pi)
        return np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)
    v = np.mod(theta + np.pi, 2 * np.pi)
    return 1. - np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)


def gii(subject, par, hemi):
    fn = (BIDS / 'derivatives' / 'prf' / 'model4' / f'sub-{subject:02d}'
          / f'sub-{subject:02d}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')
    return nib.load(fn).darrays[0].data.astype(np.float64)


def read_areas(subject, h):
    surf = FS / f'sub-{subject:02d}' / 'surf'
    va = np.round(read_morph_data(surf / f'{h}.inferred_varea')).astype(int)
    wang = np.round(np.asarray(nib.load(surf / f'{h}.wang15_mplbl.mgz').dataobj).squeeze()).astype(int)
    assert len(wang) == len(va), (len(wang), len(va))
    tri = nib.freesurfer.read_geometry(surf / f'{h}.inflated')[1]
    for code, labels in WANG.items():
        keep = despeckle(np.isin(wang, labels), tri)
        va[(va == 0) & keep] = code
    return va


def despeckle(mask, tri, n_iter=8, threshold=.3):
    """For display only: the atlas projection leaves one-vertex islands and holes."""
    n = len(mask)
    i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2], np.arange(n)])
    j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], np.arange(n)])
    a = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    a = sparse.diags(1. / np.asarray(a.sum(1)).ravel()) @ a
    m = mask.astype(float)
    for _ in range(n_iter):
        m = a @ m
    return m > threshold


def read_varea(subject):
    return np.concatenate([read_areas(subject, h) for h in ('lh', 'rh')])


def area_rgb(subject, cx_subject, border):
    va = read_varea(subject)
    curv = cortex.db.get_surfinfo(cx_subject, smooth=CURV['smooth']).data
    out = np.tile(((curv > 0) * CURV['contrast'] + CURV['brightness'])[:, None], (1, 3))
    for labels, col in AREA_COLORS.values():
        m = np.isin(va, labels)
        out[m] = .15 * out[m] + .85 * np.array(mpl.colors.to_rgb(col))
    out[border] = 1.
    return (out * 255).astype(np.uint8)


def border_mask(subject, cx_subject):
    """Vertices on the edge between visual areas (or between an area and the rest)."""
    out = []
    surfs = cortex.db.get_surf(cx_subject, 'inflated')
    for h, (_, tri) in zip(('lh', 'rh'), surfs):
        va = read_areas(subject, h)
        t = va[tri]
        mixed = ~((t[:, 0] == t[:, 1]) & (t[:, 1] == t[:, 2]))
        m = np.zeros(len(va), bool)
        m[np.unique(tri[mixed])] = True
        out.append(m)
    return np.concatenate(out)


def vertex_rgb(subject, cx_subject, par, border):
    cmap, lo, hi = PANELS[par]
    values, r2s, eccs = [], [], []
    for hemi in 'LR':
        r2, ecc = gii(subject, 'r2', hemi), gii(subject, 'ecc', hemi)
        if par == 'angle':
            v = theta_value(gii(subject, 'theta', hemi), hemi)
        elif par == 'r2':
            v = r2 / hi
        else:
            v = (gii(subject, par, hemi) - lo) / (hi - lo)
        values.append(v)
        r2s.append(r2)
        eccs.append(ecc)
    v, r2, ecc = (np.nan_to_num(np.concatenate(x)) for x in (values, r2s, eccs))
    rgb = mpl.colormaps[cmap](np.clip(v, 0, 1))[:, :3]
    alpha = np.clip((r2 - R2_RAMP[0]) / (R2_RAMP[1] - R2_RAMP[0]), 0, 1)
    if par != 'r2':
        alpha = alpha * (ecc <= ECC_MAX_SHOWN)
    curv = cortex.db.get_surfinfo(cx_subject, smooth=CURV['smooth']).data
    grey = (curv > 0) * CURV['contrast'] + CURV['brightness']
    out = alpha[:, None] * rgb + (1 - alpha[:, None]) * grey[:, None]
    out[border] = 1.
    return (out * 255).astype(np.uint8)


def headless_chrome(url):
    url = 'http://localhost:' + url.rsplit(':', 1)[1]
    profile = tempfile.mkdtemp()
    return subprocess.Popen(
        [CHROME, '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
         '--ignore-gpu-blocklist', f'--user-data-dir={profile}',
         f'--window-size={IMAGE_SIZE[0]},{IMAGE_SIZE[1]}', url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(ds, cx_subject, out_dir, stem, views, wait):
    procs = []
    webbrowser.open = lambda url, *a, **k: procs.append(headless_chrome(url)) or True
    handle = cortex.webgl.show(ds, open_browser=True, autoclose=False)
    time.sleep(wait)
    for view in views:
        fn = (out_dir / f'{stem}_{view}.png').resolve()
        if fn.exists():
            print('exists', fn, flush=True)
            continue
        # Until the surface has loaded the viewer ignores the view, silently.
        want = VIEWS[view]['camera.azimuth']
        t0 = time.time()
        while True:
            try:
                handle._set_view(**{**BASE_VIEW, **VIEWS[view]})
            except (TypeError, IndexError, KeyError):   # UI not up yet
                pass
            time.sleep(4)
            try:
                if abs(handle.ui.get('camera.azimuth')[0] - want) < 1e-3:
                    break
            except (TypeError, IndexError, KeyError):   # UI not up yet
                pass
            if time.time() - t0 > 180:
                raise TimeoutError(f'{stem} {view}: viewer did not take the view')
        time.sleep(4)
        handle.getImage(str(fn), IMAGE_SIZE)
        t0 = time.time()
        while not fn.exists():
            if time.time() - t0 > 300:
                raise TimeoutError(fn)
            time.sleep(1)
        time.sleep(2)
        print('wrote', fn, flush=True)
    handle.server.stop()
    for p in procs:
        p.terminate()


def main(subject, out_dir, views, pars, wait, r2_ramp=R2_RAMP):
    global R2_RAMP
    R2_RAMP = tuple(r2_ramp)
    store = Path(cortex.database.default_filestore)
    cx_subject = f'retsuppsub{subject:02d}'
    alias = store / cx_subject
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(alias)
    alias.symlink_to(store / f'retsupp.sub-{subject:02d}')
    try:
        cortex.db.reload_subjects()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        border = border_mask(subject, cx_subject)
        for par in pars:
            if all((out_dir / f'sub-{subject:02d}_{par}_{v}.png').exists() for v in views):
                continue
            rgb = (area_rgb(subject, cx_subject, border) if par == 'areas'
                   else vertex_rgb(subject, cx_subject, par, border))
            ds = cortex.Dataset(**{par: cortex.VertexRGB(rgb[:, 0], rgb[:, 1], rgb[:, 2],
                                                         cx_subject)})
            render(ds, cx_subject, out_dir, f'sub-{subject:02d}_{par}', views, wait)
    finally:
        alias.unlink()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--out-dir', default='notes/figures/fig1_inflated')
    ap.add_argument('--views', nargs='+', default=['pole'])
    ap.add_argument('--pars', nargs='+', default=['areas'] + list(PANELS))
    ap.add_argument('--r2-ramp', nargs=2, type=float, default=R2_RAMP,
                    help='R² at which the colour starts to show and is fully opaque')
    ap.add_argument('--wait', type=float, default=45, help='Seconds for the viewer to load')
    a = ap.parse_args()
    main(a.subject, a.out_dir, a.views, a.pars, a.wait, a.r2_ramp)
