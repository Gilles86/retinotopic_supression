"""Side analysis: build the static pycortex website of GROUP-level pRF maps on fsaverage.

Input: ``group_maps_fsaverage.npz`` from ``group_fsaverage.py`` (group-level arrays only, (2, 163842)),
copied from ``derivatives/side_prf_atlas_group/`` on the cluster to ``/data/ds-retsupp/derivatives/
side_prf_atlas_group/``. Output: a static bundle (index.html + data/) that needs no server-side code.

Runs in the ``pycortex2`` env::

    ~/mambaforge/envs/pycortex2/bin/python retsupp/side_prf_atlas/build_site.py \\
        --group-npz /data/ds-retsupp/derivatives/side_prf_atlas_group/group_maps_fsaverage.npz \\
        --annot-dir <dir with ?h.Yeo2011_7Networks_N1000.annot> \\
        --out notes/side_prf_atlas/site

The bundle uses its own pycortex subject, ``fsaverage_retsuppatlas`` (a copy of the fsaverage surfaces with a
cleaned ``overlays.svg``: the visual-area outlines of the fsaverage overlay plus a Yeo-7 default-network
outline, without the numerosity ROIs and embedded images of other projects), so the shared fsaverage cache is
never rewritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cortex
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

SRC = 'fsaverage'
SUBJ = 'fsaverage_retsuppatlas'
KEEP_ROIS = ('V1', 'V2', 'V3', 'hV4', 'V3a', 'LO', 'TO1', 'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'FEF')
INK = '{http://www.inkscape.org/namespaces/inkscape}'
SVGNS = '{http://www.w3.org/2000/svg}'
CURV = dict(curvature_brightness=0.62, curvature_contrast=0.28, curvature_smoothness=2.0, surface_specularity=0.1)


def make_subject(annot_dir):
    """Copy fsaverage into its own pycortex subject with a cleaned overlay (idempotent)."""
    store = Path(cortex.database.default_filestore)
    src, dst = store / SRC, store / SUBJ
    if (dst / 'overlays.svg').exists():
        return
    dst.mkdir()
    for d in ('surfaces', 'anatomicals', 'surface-info'):
        shutil.copytree(src / d, dst / d)
    for d in ('transforms', 'views', 'cache'):
        (dst / d).mkdir()
    for f in (dst / 'surface-info').glob('*'):
        if '[' in f.name:          # derived curvature caches are rebuilt for the new subject
            f.unlink()

    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('inkscape', 'http://www.inkscape.org/namespaces/inkscape')
    ET.register_namespace('sodipodi', 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(src / 'overlays.svg')
    root = tree.getroot()
    layers = {g.get(INK + 'label'): g for g in root if g.tag == SVGNS + 'g'}
    for c in list(layers['data']):
        layers['data'].remove(c)
    shapes = [g for g in layers['rois'] if g.get(INK + 'label') == 'shapes'][0]
    for c in list(shapes):
        if c.get(INK + 'label') not in KEEP_ROIS:
            shapes.remove(c)
    for g in list(layers['rois']):
        if g.get(INK + 'label') == 'labels':
            for c in list(g):
                g.remove(c)

    # Default network outline (Yeo 2011, 7 networks, network 7), contoured on the flat map.
    w, h = float(root.get('width')), float(root.get('height'))
    pts, _ = cortex.db.get_surf(SUBJ, 'flat', merge=True, nudge=True)
    c = pts[:, :2].astype(float)
    c -= c.min(0)
    c /= c.max(0)
    xy = c * np.array([w, h])
    nl = len(cortex.db.get_surf(SUBJ, 'flat', hemisphere='lh')[0])
    dmn = np.concatenate([nib.freesurfer.read_annot(str(Path(annot_dir) / f'{fh}.Yeo2011_7Networks_N1000.annot'))[0] == 7
                          for fh in ('lh', 'rh')])
    grp = ET.SubElement(shapes, SVGNS + 'g', {INK + 'label': 'DMN', 'id': 'roi_DMN'})
    for hs in (slice(0, nl), slice(nl, None)):
        hxy, hm = xy[hs], dmn[hs]
        lo, hi = hxy.min(0), hxy.max(0)
        nx = 900
        ny = int(nx * (hi[1] - lo[1]) / (hi[0] - lo[0]))
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
        dist, nn = cKDTree(hxy).query(np.c_[gx.ravel(), gy.ravel()])
        img = np.where(dist < 3 * (hi[0] - lo[0]) / nx, hm[nn], 0).reshape(ny, nx).astype(float)
        cs = plt.contour(gx, gy, gaussian_filter(img, 2), levels=[.5])
        segs = [s for s in cs.allsegs[0] if len(s) > 10]
        plt.close('all')
        area = [0.5 * abs(np.dot(s[:, 0], np.roll(s[:, 1], 1)) - np.dot(s[:, 1], np.roll(s[:, 0], 1))) for s in segs]
        segs = [s for s, a in zip(segs, area) if a >= 0.05 * max(area)]
        for s in segs:
            d = 'M ' + ' L '.join(f'{x:.1f},{h - y:.1f}' for x, y in s) + ' Z'
            ET.SubElement(grp, SVGNS + 'path', {'d': d, 'style': 'fill:none;stroke:#ffffff;stroke-width:2'})
    tree.write(dst / 'overlays.svg', xml_declaration=True, encoding='utf-8')
    ov = cortex.db.get_overlay(SUBJ, modify_svg_file=False)
    got = np.zeros(len(dmn), bool)
    got[np.asarray(ov.rois.get_mask('DMN'), dtype=int)] = True
    dice = 2 * (got & dmn).sum() / (got.sum() + dmn.sum())
    print(f'DMN outline written; Dice vs Yeo network 7 = {dice:.3f}; ROIs: {sorted(ov.rois.shapes)}')


def vtx(G, key):
    return np.concatenate([G[key][0], G[key][1]]).astype(float)


def layer(values, alpha, cmap, vmin, vmax):
    v = np.where(np.isfinite(values), values, vmin)
    a = np.where(np.isfinite(values) & np.isfinite(alpha), alpha, 0.0).clip(0, 1) * 0.92
    return cortex.Vertex(v, SUBJ, vmin=vmin, vmax=vmax, cmap=cmap).blend_curvature(a, brightness=0.62, contrast=0.28,
                                                                                   smooth=2)


def gradient_css(cmap, n=24):
    cols = plt.get_cmap(cmap)(np.linspace(0, 1, n))[:, :3]
    return 'linear-gradient(to right, ' + ', '.join('rgb(%d,%d,%d)' % tuple(int(255 * c) for c in r) for r in cols) + ')'


def build(G, out):
    ok = vtx(G, 'n_subjects') >= 10
    rel_mask = vtx(G, 'rel_mask') > 0
    cv_mask = vtx(G, 'cv_mask') > 0
    neg_mask = vtx(G, 'neg_mask') > 0
    angle_ok = rel_mask & (vtx(G, 'm4_angle_rayleigh_p') < .05)
    par_ok = rel_mask & (vtx(G, 'm4_n_used') >= 5)
    neg_angle_ok = neg_mask & (vtx(G, 'neg_angle_rayleigh_p') < .05)
    deg = 180 / np.pi
    L = [  # name, values, alpha, cmap, vmin, vmax, unit label, description
        ('Polar angle', vtx(G, 'm4_polar_angle') * deg, angle_ok, 'hsv', -180, 180, 'deg (0 = right, 90 = up)',
         'Circular mean over the subjects with a reliable response at the vertex; shown where those angles agree (Rayleigh p < .05).'),
        ('Eccentricity', vtx(G, 'm4_ecc'), par_ok, 'viridis', 0, 4, 'deg',
         'Median over subjects with a reliable response. The bar aperture had a radius of 3.17 deg.'),
        ('pRF size (sigma)', vtx(G, 'm4_sd'), par_ok, 'plasma', 0, 3, 'deg', 'Median model-4 centre sigma.'),
        ('HRF peak delay', vtx(G, 'm4_hrf_delay'), par_ok, 'cividis', 3.5, 6.5, 's', 'Median flexible-HRF delay of model 4.'),
        ('Reliable response (% subjects)', vtx(G, 'rel_prev') * 100, rel_mask.astype(float), 'inferno', 0, 80, '% of subjects',
         'Share of subjects whose cross-run reliability beats a circular-shift null (p < .05). Shown where that share exceeds chance (binomial vs 5%, FDR q < .01).'),
        ('Cross-validated pRF (% subjects)', vtx(G, 'cv_prev') * 100, cv_mask.astype(float), 'inferno', 0, 80, '% of subjects',
         'Share of subjects whose signed Gaussian pRF, fitted on one session, predicts the other session better than with a shifted stimulus (p < .05). Masked as above.'),
        ('Signed pRF: positive vs negative', vtx(G, 'sign_balance'), cv_mask.astype(float), 'RdBu_r', -1, 1, '(pos - neg) / (pos + neg)',
         'Among subjects with a cross-validated pRF whose sign agrees in both sessions: +1 all positive, -1 all negative.'),
        ('Negative pRFs (% subjects)', vtx(G, 'neg_prev') * 100, neg_mask.astype(float), 'Blues', 0, 40, '% of subjects',
         'Share of subjects with a cross-validated negative pRF (negative in the all-runs fit and in both session fits). Shown where above chance (binomial vs 2.5%, FDR q < .01).'),
        ('Negative pRFs: polar angle', vtx(G, 'neg_polar_angle') * deg, neg_angle_ok, 'hsv', -180, 180, 'deg',
         'Circular mean polar angle of the negative pRFs; shown where they agree across subjects (Rayleigh p < .05).'),
        ('Model-4 R2 (median, in-sample)', vtx(G, 'median_m4_r2'), ok.astype(float), 'magma', 0, 0.3, 'R2',
         'Median in-sample R2 of the model-4 fit over all subjects, unmasked.'),
    ]
    ds, legend = {}, {}
    for name, v, a, cmap, lo, hi, unit, desc in L:
        n_shown = int((np.asarray(a, float) > 0).sum())
        print(f'{name}: {n_shown} vertices shown, range {np.nanmin(v[a > 0]) if n_shown else np.nan:.3g}..'
              f'{np.nanmax(v[a > 0]) if n_shown else np.nan:.3g}')
        if n_shown == 0:
            continue
        ds[name] = layer(v, np.asarray(a, float), cmap, lo, hi)
        legend[name] = dict(cmap=gradient_css(cmap), vmin=f'{lo:g}', vmax=f'{hi:g}', label=unit, desc=desc)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'data').exists():
        shutil.rmtree(out / 'data')
    for f in out.glob('*'):
        if f.is_file():
            f.unlink()
    cortex.webgl.make_static(str(out), ds, types=('inflated',), recache=False, title='retsupp pRF atlas', **CURV)
    inject(out / 'index.html', legend, int(len(G['subjects'])))
    return ds


def inject(index_html, legend, n_subj):
    html = index_html.read_text()
    panel = """
<style>
#atl-info, #atl-legend { position: fixed; z-index: 10000; font: 14px/1.45 -apple-system, system-ui, sans-serif;
  color: #eee; background: rgba(18,18,18,.9); border: 1px solid #4a4a4a; border-radius: 10px; padding: 12px 16px; }
#atl-info { left: 16px; top: 16px; max-width: 420px; }
#atl-info h1 { font-size: 17px; margin: 0 0 6px; }
#atl-info p { margin: 4px 0; color: #ccc; }
#atl-info details summary { cursor: pointer; color: #9cc3ff; }
#atl-legend { right: 16px; bottom: 16px; width: min(440px, calc(100vw - 64px)); }
#atl-legend .nm { font-weight: 600; margin-bottom: 6px; }
#atl-legend .bar { height: 22px; border-radius: 4px; border: 1px solid #666; }
#atl-legend .lim { display: flex; justify-content: space-between; color: #bbb; font-size: 13px; margin-top: 4px; }
#atl-legend .ds { color: #bbb; font-size: 13px; margin-top: 6px; }
@media (max-width: 700px) { #atl-info { max-width: calc(100vw - 64px); } #atl-info details { display: none; } }
</style>
<div id="atl-info"><h1>retsupp pRF atlas (fsaverage)</h1>
<p>Group maps from __N__ participants, 7T bar pRF mapping during a visual search task. No individual maps.
Pick a map in the viewer's data menu; white outlines are visual areas (fsaverage overlay) and the default network (Yeo 2011, network 7).</p>
<details><summary>Methods and masks</summary>
<p>pRF parameters (polar angle, eccentricity, size, HRF delay) come from a difference-of-Gaussians pRF model with a flexible HRF,
fitted to all runs per participant in volume space and resampled to fsaverage (nearest voxel at mid-thickness, nearest vertex on the registered sphere).</p>
<p>Reliability does not use any pRF model: per participant and vertex, the correlation of each run with the mean of the other runs must beat a
null in which that mean is circularly shifted (100 shifts, p &lt; .05). A vertex enters the group maps when more participants pass than expected by chance
(binomial test against 5%, Benjamini-Hochberg q &lt; .01), and each parameter is summarised only over the participants that pass at that vertex (at least 5).</p>
<p>Signed pRFs: a Gaussian pRF grid fit that allows negative amplitudes, fitted on one session and tested on the other (both directions),
against 50 fits with a circularly shifted stimulus.</p></details></div>
<div id="atl-legend"><div id="atl-body"></div></div>
<script>
(function () {
  var CB = __ENTRIES__;
  function show(name) {
    if (name instanceof Array) name = name[0];
    var e = CB[name] || CB[Object.keys(CB)[0]], k = CB[name] ? name : Object.keys(CB)[0];
    document.getElementById('atl-body').innerHTML = '<div class="nm">' + k + '</div><div class="bar" style="background:' + e.cmap +
      '"></div><div class="lim"><span>' + e.vmin + '</span><span>' + e.label + '</span><span>' + e.vmax + '</span></div><div class="ds">' + e.desc + '</div>';
  }
  document.addEventListener('DOMContentLoaded', function () { show(null); });
  var tries = 0, iv = setInterval(function () {
    if (typeof mriview !== 'undefined' && mriview.Viewer && mriview.Viewer.prototype.setData) {
      clearInterval(iv);
      var orig = mriview.Viewer.prototype.setData;
      mriview.Viewer.prototype.setData = function (name) {
        var r = orig.apply(this, arguments);
        try { show(this.active ? this.active.name : name); } catch (err) { show(name); }
        return r;
      };
      try { if (typeof figure !== 'undefined') for (var q in figure) { var c = figure[q]; if (c && c.active && c.active.name) { show(c.active.name); break; } } } catch (err) {}
    } else if (++tries > 100) clearInterval(iv);
  }, 100);
})();
</script>
"""
    panel = panel.replace('__ENTRIES__', json.dumps(legend)).replace('__N__', str(n_subj))
    index_html.write_text(html + panel)


def render_pngs(ds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in ds.items():
        fig = cortex.quickflat.make_figure(d, with_curvature=False, with_rois=True, with_sulci=False,
                                           with_labels=False, with_colorbar=False, height=1024)
        fn = out_dir / (name.replace(' ', '_').replace('/', '').replace(':', '').replace('(', '').replace(')', '')
                        .replace('%', 'pct').replace(',', '') + '.png')
        fig.savefig(fn, dpi=100, facecolor='k')
        plt.close(fig)
        print(f'  {fn}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group-npz', required=True)
    ap.add_argument('--annot-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--png-dir', default=None, help='also render flat-map PNGs here, for checking')
    a = ap.parse_args()
    make_subject(a.annot_dir)
    G = dict(np.load(a.group_npz))
    ds = build(G, a.out)
    if a.png_dir:
        render_pngs(ds, a.png_dir)
    size = sum(f.stat().st_size for f in Path(a.out).rglob('*') if f.is_file())
    print(f'site: {a.out} ({size / 1e6:.1f} MB)')
