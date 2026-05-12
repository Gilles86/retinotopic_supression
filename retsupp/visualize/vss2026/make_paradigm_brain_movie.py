"""3-panel movie: paradigm | LH inflated lateral | RH inflated lateral.

For each subject we average the cleaned BOLD across all 12 runs
(2 sessions × 6 runs, with a couple of subjects missing one run), z-score
per vertex along time, and project it onto the fsnative inflated surface.
Each hemisphere is rendered as a 2D rasterised orthographic image with
the camera at a slightly-lateral angle behind the brain, so the visual
cortex is visible at the back of the panel. Rendering uses a pre-computed
pixel→vertex lookup, so each frame is just a numpy fancy-index — no
per-frame matplotlib 3-D redraw — which makes the movie cheap enough to
loop over several subjects in one go.

The paradigm panel paints a small faithful schematic of the on-screen
display: black background, gray aperture circle, a white sweeping bar at
the run-deterministic position, a small fixation cross at the centre.
(Search-array items are not rendered — they only flash on for ~1.5 s
per trial and would clutter the bar sweep.)

Per-run vol-to-surf is cached under
``~/.cache/retsupp/surfbold/sub-XX_ses-Y_run-Z_hemi-?.npy`` so subsequent
re-renders only pay the matplotlib cost.

Example::

    ~/mambaforge/envs/retsupp/bin/python -m \\
        retsupp.visualize.vss2026.make_paradigm_brain_movie \\
        --subjects 2 3 5 \\
        --out-dir notes/figures/talk
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as manim
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Normalize
from nilearn import surface
from nilearn.plotting import plot_surf_stat_map
from scipy.spatial import cKDTree

from retsupp.utils.data import Subject


PRES_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
}

CACHE_DIR = Path.home() / ".cache" / "retsupp" / "surfbold"
N_VOLUMES = 258


# ---------------------------------------------------------------------------
# BOLD on the surface, averaged across runs
# ---------------------------------------------------------------------------

def _vol_to_surf(bold_img, inner, outer):
    return surface.vol_to_surf(
        bold_img,
        surf_mesh=str(outer),
        inner_mesh=str(inner),
        interpolation="linear",
    ).astype(np.float32)


def _zscore(ts, axis=1):
    mu = np.nanmean(ts, axis=axis, keepdims=True)
    sd = np.nanstd(ts, axis=axis, keepdims=True)
    sd[sd == 0] = 1.0
    return ((ts - mu) / sd).astype(np.float32)


def _psc(ts, axis=1):
    """Percent signal change relative to each vertex's run-mean. Vertices
    with a near-zero mean (mostly outside the brain) are returned as 0.
    """
    mu = np.nanmean(ts, axis=axis, keepdims=True)
    safe_mu = np.where(np.abs(mu) < 1e-3, 1.0, mu)
    out = (ts - mu) / safe_mu * 100.0
    out = np.where(np.abs(mu) < 1e-3, 0.0, out)
    return out.astype(np.float32)


def _cached_surf_bold(subject: Subject, session: int, run: int,
                      cache_dir: Path = CACHE_DIR):
    """Return ``{'L': (V_L, T), 'R': (V_R, T)}`` arrays of cleaned BOLD
    projected onto fsnative; cached per (subject, ses, run) on disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    paths = {h: cache_dir / (
        f"sub-{subject.subject_id:02d}_ses-{session}_run-{run}_hemi-{h}.npy"
    ) for h in "LR"}
    if all(p.exists() for p in paths.values()):
        return {h: np.load(p) for h, p in paths.items()}

    bold = subject.get_bold(session=session, run=run, type="cleaned")
    surf_info = subject.get_surf_info()
    for h in "LR":
        ts = _vol_to_surf(
            bold,
            inner=surf_info[h]["inner"],
            outer=surf_info[h]["outer"],
        )
        # Crop to N_VOLUMES (cleaned BOLD sometimes 259)
        ts = ts[:, :N_VOLUMES]
        np.save(paths[h], ts)
        out[h] = ts
    return out


def average_bold_across_runs(subject: Subject,
                              *, cache_dir: Path = CACHE_DIR,
                              signal: str = "psc",
                              verbose: bool = True):
    """Compute per-(V, T) mean, SD, and t-statistic across runs for each
    hemisphere. Returns ``{'mean', 'sd', 't', 'n_runs'}``.

    Cleaned BOLD on disk is **already PSC** (clean.py emits PSC), so we
    use it directly — no further transformation. The per-(vertex, TR)
    t-stat tests whether the mean PSC across runs is reliably different
    from zero. df = n_runs − 1.

    ``signal`` is accepted for backward-compat but currently only 'psc'
    is meaningful for the cached data.
    """
    sum1 = {"L": None, "R": None}
    sum2 = {"L": None, "R": None}
    n = 0
    for ses in (1, 2):
        runs = subject.get_runs(session=ses)
        for run in runs:
            if verbose:
                print(f"  ses-{ses} run-{run}…", flush=True)
            ts = _cached_surf_bold(subject, ses, run, cache_dir=cache_dir)
            for h in "LR":
                x = ts[h].astype(np.float32)
                sum1[h] = x if sum1[h] is None else sum1[h] + x
                sum2[h] = x * x if sum2[h] is None else sum2[h] + x * x
            n += 1
    if verbose:
        print(f"  averaged {n} runs (PSC, cleaned/PSC on disk) + t-stat (df={n - 1})")
    out_mean = {}
    out_sd = {}
    out_t = {}
    for h in "LR":
        mean = (sum1[h] / n).astype(np.float32)
        var = (sum2[h] - n * mean * mean) / max(n - 1, 1)
        var = np.maximum(var, 0.0)
        sd = np.sqrt(var).astype(np.float32)
        sem = sd / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sem > 1e-9, mean / sem, 0.0).astype(np.float32)
        out_mean[h] = mean
        out_sd[h] = sd
        out_t[h] = t
    return {"mean": out_mean, "sd": out_sd, "t": out_t, "n_runs": n}


# ---------------------------------------------------------------------------
# Surface assets: inflated mesh, curvature, R² map
# ---------------------------------------------------------------------------

def _load_r2_surf(subject: Subject, model: int = 4):
    base = (
        Path(subject.bids_folder)
        / "derivatives" / "prf" / f"model{model}"
        / f"sub-{subject.subject_id:02d}"
    )
    out = {}
    for hemi in "LR":
        fn = base / (
            f"sub-{subject.subject_id:02d}_desc-r2.optim.nilearn"
            f"_space-fsnative_hemi-{hemi}.func.gii"
        )
        out[hemi] = nib.load(fn).darrays[0].data.astype(np.float32)
    return out


def _load_inflated(subject: Subject):
    """Load the fsnative inflated mesh and the (much smoother) sulcal-depth
    map for each hemi; sulc gives clean anatomical landmarks without the
    high-spatial-frequency noise of mean curvature.
    """
    fs = (
        Path(subject.bids_folder)
        / "derivatives" / "freesurfer" / f"sub-{subject.subject_id:02d}"
    )
    out = {}
    for hemi, fs_hemi in (("L", "lh"), ("R", "rh")):
        infl = surface.load_surf_mesh(str(fs / "surf" / f"{fs_hemi}.inflated"))
        sulc = nib.freesurfer.read_morph_data(
            str(fs / "surf" / f"{fs_hemi}.sulc")
        )
        out[hemi] = {"mesh": infl, "curv": sulc}
    return out


N_PARAMS_BY_MODEL = {1: 5, 2: 7, 3: 7, 4: 9, 6: 11,
                     7: 9, 8: 9, 9: 9, 10: 9, 11: 9}


def _fdr_r2_threshold(r2_all: np.ndarray, *, alpha: float = 0.001,
                       n: int = N_VOLUMES, k: int = 7) -> float:
    """Return the smallest R² that survives BH-FDR at level ``alpha`` on the
    F-test p-values for k-parameter PRF fits over n TRs.
    """
    from scipy.stats import f as f_dist
    from statsmodels.stats.multitest import multipletests
    r2c = np.clip(r2_all, 0.0, 0.999999)
    df1, df2 = k, n - k - 1
    F = (r2c / df1) / ((1.0 - r2c) / df2)
    p = 1.0 - f_dist.cdf(F, df1, df2)
    finite = np.isfinite(r2c)
    rejected, *_ = multipletests(p[finite], alpha=alpha, method="fdr_bh")
    if not rejected.any():
        return np.inf
    return float(np.min(r2c[finite][rejected]))


# ---------------------------------------------------------------------------
# Per-hemi 2D orthographic projection and pixel→vertex rasterisation
# ---------------------------------------------------------------------------

def _compute_vertex_normals(coords3d, faces):
    """Mean of incident face normals, normalised. Used for Lambertian
    shading so the inflated mesh has proper 3-D depth cues."""
    v0 = coords3d[faces[:, 0]]
    v1 = coords3d[faces[:, 1]]
    v2 = coords3d[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-12)
    vn = np.zeros_like(coords3d)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-12)
    return vn


def _project_hemi(coords, faces, *, side: str, lateral_angle_deg: float,
                   view: str = "lateral"):
    """Project a fsnative hemisphere mesh onto a 2D panel from a slightly
    side-tilted camera angle behind the brain.

    Parameters
    ----------
    side : 'L' or 'R'
        Which hemisphere; controls which side the camera rotates toward.
    lateral_angle_deg : float
        0 = pure posterior view; 90 = pure side view (lateral or medial).
        Typical: 20–30°.
    view : {'lateral', 'medial'}
        For 'lateral' the camera looks at the outside of the hemisphere
        (LH from -x, RH from +x). For 'medial' the camera looks at the
        inside (LH from +x, RH from -x) — the calcarine V1 region is best
        seen in this view, since on the inflated mesh V1 sits on the
        medial-posterior surface.

    Returns
    -------
    coords_xy : (V, 2) float32
        Projected coordinates in the panel plane.
    faces_visible : (F', 3) int
        Face indices, culled to camera-facing only.
    """
    theta = np.deg2rad(lateral_angle_deg)
    sign = -1.0 if side == "L" else 1.0
    if view == "medial":
        sign = -sign
    elif view != "lateral":
        raise ValueError(f"view must be 'lateral' or 'medial', got {view!r}")
    cam = np.array([sign * np.sin(theta), -np.cos(theta), 0.0])  # posn from origin
    view = -cam  # camera looks toward origin
    up = np.array([0.0, 0.0, 1.0])
    # right = view × up so that anatomical convention is preserved:
    #   LH lateral view → anterior on the LEFT of the LH panel
    #   RH lateral view → anterior on the RIGHT of the RH panel
    right = np.cross(view, up)
    right = right / np.linalg.norm(right)
    x = coords @ right
    y = coords @ up
    coords_xy = np.stack([x, y], axis=1).astype(np.float32)

    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    n_norms = np.linalg.norm(normals, axis=-1)
    cos_angle = (normals @ cam) / np.maximum(n_norms, 1e-12)
    # Drop both back-facing AND near-edge-on faces (projected area ≈ 0,
    # which makes matplotlib's TriFinder fail with "invalid triangulation").
    visible = cos_angle > 0.05
    faces_v = faces[visible]

    # Belt-and-suspenders: drop faces whose projected 2D triangle is
    # numerically zero-area (can still happen after culling for some
    # near-degenerate inflated-mesh faces).
    tri_xy = coords_xy[faces_v]
    area2 = (
        (tri_xy[:, 1, 0] - tri_xy[:, 0, 0])
        * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
        - (tri_xy[:, 2, 0] - tri_xy[:, 0, 0])
        * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1])
    )
    faces_v = faces_v[np.abs(area2) > 1e-4]
    return coords_xy, faces_v


def _build_pixel_lookup(coords_xy, faces, *, width: int, height: int,
                         knn: int = 12, smoothing_sigma_mult: float = 2.5):
    """Rasterise a projected mesh to (height, width) using inverse-Gauss-
    weighted k-NN over the vertices used by ``faces`` (the visible
    front-facing surface). Returns ``(pix_to_vert_knn, pix_weight, inside,
    extent)`` where ``pix_to_vert_knn`` is (H, W, knn) of global vertex
    indices and ``pix_weight`` is (H, W, knn) of normalised weights.

    A KDTree-based lookup is used because matplotlib's TrapezoidMapTriFinder
    rejects projected brain meshes (overlapping front/back triangles).
    """
    used = np.unique(faces.flatten())
    sub_xy = coords_xy[used]

    x_min, x_max = sub_xy[:, 0].min(), sub_xy[:, 0].max()
    y_min, y_max = sub_xy[:, 1].min(), sub_xy[:, 1].max()
    pad = 0.04
    dx = (x_max - x_min); dy = (y_max - y_min)
    x_min -= pad * dx; x_max += pad * dx
    y_min -= pad * dy; y_max += pad * dy

    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_min, y_max, height)
    XX, YY = np.meshgrid(xs, ys)
    pix_xy = np.stack([XX, YY], axis=-1).reshape(-1, 2)

    tree = cKDTree(sub_xy)
    dists, idx = tree.query(pix_xy, k=knn)  # (H*W, knn)

    # Typical nearest-neighbour vertex spacing — sets Gauss sigma + inside
    # mask threshold.
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(sub_xy.shape[0],
                            size=min(2000, sub_xy.shape[0]),
                            replace=False)
    sample_tree = cKDTree(sub_xy[sample_idx])
    nn_d, _ = sample_tree.query(sub_xy[sample_idx], k=2)
    typical = float(np.median(nn_d[:, 1]))
    sigma = typical * smoothing_sigma_mult

    weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1e-12)
    weights = weights.astype(np.float32)

    pix_to_vert_knn = used[idx].astype(np.int64)

    thresh = max(typical * 4.0, dx / width * 1.5)
    inside = (dists[:, 0] < thresh).reshape(XX.shape)
    H, W = XX.shape
    pix_to_vert_knn = pix_to_vert_knn.reshape(H, W, knn)
    weights = weights.reshape(H, W, knn)
    weights[~inside] = 0.0  # outside pixels contribute zero

    return pix_to_vert_knn, weights, inside, (x_min, x_max, y_min, y_max)


def _alpha_from_r2(r2, lo: float, hi: float):
    return np.clip((r2 - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _alpha_binary_fdr(r2, threshold: float):
    return (r2 >= threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Brain panel: per-vertex RGBA → image lookup
# ---------------------------------------------------------------------------

def _make_brain_renderer(coords_xy, faces, curv, *,
                          width: int, height: int,
                          stat_vmin: float, stat_vmax: float,
                          bg_darkness: float = 0.55,
                          knn: int = 12,
                          vertex_normals: np.ndarray | None = None,
                          light_dir: np.ndarray | None = None,
                          cmap_name: str = "magma",
                          t_cutoff: float = 3.1):
    """Build a per-frame brain renderer. The rasteriser averages each pixel
    over its ``knn`` nearest projected vertices with inverse-Gauss weights
    (gentle screen-space smoothing). If ``vertex_normals`` + ``light_dir``
    are provided, Lambertian shading is baked into the curvature
    background — this gives the inflated mesh proper 3-D depth cues
    without the cost of matplotlib 3-D rendering.
    """
    pix_to_vert_knn, weights, inside, extent = _build_pixel_lookup(
        coords_xy, faces, width=width, height=height, knn=knn,
    )

    # Sulc gives smooth anatomical context — compress to a narrow gray band.
    sulc_q = np.quantile(curv[np.isfinite(curv)], [0.1, 0.9])
    bg_norm = Normalize(vmin=float(sulc_q[0]), vmax=float(sulc_q[1]))
    bg_vert = plt.get_cmap("Greys_r")(bg_norm(curv))
    bg_vert[:, :3] = 0.45 + 0.45 * bg_vert[:, :3]  # base gray band
    bg_vert[:, 3] = 1.0
    bg_vert = bg_vert.astype(np.float32)

    # Lambertian per-vertex shading: 0.35 (ambient) + 0.65 * max(0, n·light).
    # Light is "headlamp" from camera by default.
    if vertex_normals is not None and light_dir is not None:
        lambert = np.maximum(0.0, vertex_normals @ light_dir).astype(np.float32)
        shade = 0.35 + 0.65 * lambert  # (V,)
        bg_vert[:, :3] *= shade[:, None]

    stat_cmap = plt.get_cmap(cmap_name)
    stat_norm = Normalize(vmin=stat_vmin, vmax=stat_vmax)

    # Static background image (curvature × Lambertian shading).
    bg_img = (bg_vert[pix_to_vert_knn] * weights[..., None]).sum(axis=-2)
    bg_img[:, :, 3] = 1.0
    outside = ~inside
    bg_img[outside] = 1.0

    inside_col = inside[..., None].astype(np.float32)

    def render(z_vert, t_vert):
        """Render one frame given per-vertex mean PSC ``z_vert`` and per-
        vertex t-statistic ``t_vert``. Pixels where t < t_cutoff (signed)
        are fully transparent; pixels in [t_cutoff, t_cutoff*1.4] ramp in
        opacity for soft edges. Only POSITIVE activations are shown —
        deactivations (negative t) are treated as background.
        """
        z_pix = (z_vert[pix_to_vert_knn] * weights).sum(axis=-1)
        t_pix = (t_vert[pix_to_vert_knn] * weights).sum(axis=-1)
        ramp_lo, ramp_hi = t_cutoff, t_cutoff * 1.4
        t_alpha = np.clip((t_pix - ramp_lo) / (ramp_hi - ramp_lo), 0.0, 1.0)
        combined_a = (t_alpha * inside)[..., None]
        fg = stat_cmap(stat_norm(z_pix)).astype(np.float32)
        out = combined_a * fg + (1.0 - combined_a) * bg_img
        out[:, :, 3] = 1.0
        return out

    return render, extent


# ---------------------------------------------------------------------------
# Paradigm panel: black canvas with aperture, sweeping bar, fixation
# ---------------------------------------------------------------------------

def _make_checkerboard(n_cells_short: int = 8, n_cells_long: int = 64,
                        upsample: int = 4):
    """Return a (H, W) float32 checkerboard tile with values in {0, 1}, with
    enough cells along the long axis to span the aperture diameter.
    """
    base = (np.indices((n_cells_long, n_cells_short)).sum(axis=0) % 2).astype(
        np.float32
    )
    return np.kron(base, np.ones((upsample, upsample), dtype=np.float32))


def _paradigm_artists(ax, settings, *, half_extent: float):
    """Create static paradigm artists (aperture circle, fixation cross) and
    one updatable checkerboard bar (as AxesImage, clipped to the aperture
    circle). Returns ``(bar_image, aperture_circle, checker_tiles)`` where
    the caller updates the bar per frame and can flip the tile pair for
    the 8 Hz flicker.
    """
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(-half_extent, half_extent)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("black")

    rad = settings["radius_bar_aperture"]
    bar_width = settings["bar_width"]

    aperture = mpatches.Circle(
        (0, 0), rad, edgecolor="0.55", facecolor="none", linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(aperture)

    # Two checkerboard tiles with inverted phase to give an 8 Hz-style flicker
    tile_a = _make_checkerboard()
    tile_b = 1.0 - tile_a
    # Horizontal sweep: bar is vertical strip of width `bar_width` and full
    # diameter height. We use one imshow whose extent we update per frame.
    bar_img = ax.imshow(
        tile_a,
        extent=(-bar_width / 2, bar_width / 2, -rad, rad),
        origin="lower",
        cmap="gray", vmin=0.0, vmax=1.0,
        interpolation="nearest",
        zorder=2,
        visible=False,
    )
    # Clip the bar to the circular aperture so it never extends past the
    # edge of the FOV (which it would otherwise do during the sweep).
    bar_img.set_clip_path(aperture)

    # Fixation: small dot + cross
    fix_len = 0.30
    ax.plot([-fix_len, fix_len], [0, 0],
            color="0.85", lw=1.6, zorder=5)
    ax.plot([0, 0], [-fix_len, fix_len],
            color="0.85", lw=1.6, zorder=5)
    ax.plot(0, 0, "o", mfc="0.65", mec="none", markersize=4, zorder=6)

    return bar_img, aperture, (tile_a, tile_b)


def _bar_state_per_TR(subject: Subject, sessions=(1,), runs=(1,),
                      settings=None, n_volumes: int = N_VOLUMES,
                      upsample: int = 1):
    """Compute (orientation_deg, position_along_sweep_axis, visible_bool)
    per displayed frame for the first available (session, run). The bar
    sequence is deterministic so any run's events.tsv works.

    ``upsample`` produces sub-TR temporal resolution by sampling the bar's
    continuous trajectory at frame_times = linspace(TR/2, ..., n_frames)
    instead of just at TR centres. n_frames = (n_volumes - 1) * upsample
    + 1.
    """
    ses, run = sessions[0], runs[0]
    onsets = subject.get_onsets(ses, run)
    tr = subject.get_tr(ses, run)
    if upsample == 1:
        n_frames = n_volumes
        frametimes = np.arange(tr / 2.0, tr * n_volumes + tr / 2.0, tr)
    else:
        n_frames = (n_volumes - 1) * upsample + 1
        t_start = tr / 2.0
        t_end = tr / 2.0 + (n_volumes - 1) * tr
        frametimes = np.linspace(t_start, t_end, n_frames)

    bar_events = onsets[onsets["event_type"].str.startswith("bar")]
    rad = settings["radius_bar_aperture"]
    bar_w = settings["bar_width"]
    speed = settings["speed"]

    ori = np.zeros(n_frames, dtype=np.float32)
    pos = np.full(n_frames, np.nan, dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)

    for i, t in enumerate(frametimes):
        prev = bar_events[bar_events["onset"] < t]
        if len(prev) == 0:
            continue
        last = prev.iloc[-1]
        state = last["event_type"]
        dt = t - last["onset"]
        if state in ("bar_rest", "bar_break"):
            continue
        if state == "bar_right":
            pos[i] = -rad - bar_w / 2 + dt * speed
            ori[i] = 0
        elif state == "bar_left":
            pos[i] = rad + bar_w / 2 - dt * speed
            ori[i] = 0
        elif state == "bar_up":
            pos[i] = -rad - bar_w / 2 + dt * speed
            ori[i] = 90
        elif state == "bar_down":
            pos[i] = rad + bar_w / 2 - dt * speed
            ori[i] = 90
        else:
            continue
        if -rad - bar_w / 2 <= pos[i] <= rad + bar_w / 2:
            visible[i] = True

    return ori, pos, visible


def _upsample_temporal(arr_2d, upsample: int):
    """Linearly interpolate a (V, T) array along the time axis to
    (V, (T-1)*upsample + 1). Vectorised: two slice multiplies, one add.
    """
    if upsample == 1:
        return arr_2d
    V, T = arr_2d.shape
    new_T = (T - 1) * upsample + 1
    new_idx = np.linspace(0, T - 1, new_T)
    t_lo = np.floor(new_idx).astype(np.int64)
    t_hi = np.minimum(t_lo + 1, T - 1)
    frac = (new_idx - t_lo).astype(arr_2d.dtype)
    out = arr_2d[:, t_lo] * (1.0 - frac) + arr_2d[:, t_hi] * frac
    return out.astype(arr_2d.dtype)


def _update_bar(bar_img, ori_deg, pos, *, bar_width, rad, visible,
                tiles, frame_idx):
    """Place the checkerboard bar at the right position/orientation and
    flicker the texture by swapping tile_a ↔ tile_b every frame.
    """
    if not visible:
        bar_img.set_visible(False)
        return
    bar_img.set_visible(True)
    tile = tiles[frame_idx % 2]
    # Horizontal sweep (ori=0): vertical bar of width=bar_width, height=2*rad
    # centred at (pos, 0). Vertical sweep (ori=90): horizontal bar of
    # width=2*rad, height=bar_width, centred at (0, pos).
    if ori_deg == 0:
        extent = (pos - bar_width / 2, pos + bar_width / 2, -rad, rad)
        bar_img.set_data(tile)
    else:
        extent = (-rad, rad, pos - bar_width / 2, pos + bar_width / 2)
        # Transpose so the checker bands run along the bar's long axis
        bar_img.set_data(tile.T)
    bar_img.set_extent(extent)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_subject(
    subject_id: int,
    *,
    bids_folder: Path,
    out_path: Path,
    fps: int = 8,
    model: int = 1,
    fdr_alpha: float = 0.001,
    lateral_angle_deg: float = 25.0,
    brain_width: int = 520,
    brain_height: int = 480,
    paradigm_resolution: int = 200,  # unused now; kept for CLI compatibility
    max_frames: int | None = None,
    dpi: int = 110,
    signal: str = "psc",
    signal_clip: float = 2.0,
    normalize: str = "absolute",
    view: str = "lateral",
    hrf_lag_trs: int = 3,
    t_cutoff: float = 3.1,
    temporal_upsample: int = 4,
):
    subject = Subject(subject_id, bids_folder=str(bids_folder))

    # Bar geometry from session 1 run 1 (deterministic sequence)
    settings = subject.get_experimental_settings(session=1, run=1)
    rad = float(settings["radius_bar_aperture"])
    bar_w = float(settings["bar_width"])
    ring = float(settings["eccentricity_stimuli"])
    half_extent = max(rad, ring + float(settings["size_stimuli"])) * 1.12

    print(
        f"[sub-{subject_id:02d}] computing per-(vertex, TR) mean+sd+t "
        f"across runs ({signal})"
    )
    stats = average_bold_across_runs(subject, signal=signal)
    bold_mean = stats["mean"]
    bold_t = stats["t"]
    n_runs = stats["n_runs"]
    print(
        f"[sub-{subject_id:02d}] {n_runs} runs averaged; "
        f"per-TR t-cutoff |t| > {t_cutoff:.2f} (df={n_runs - 1}, ~α=0.01)"
    )

    # Shift mean+t back by hrf_lag_trs so that frame N shows the brain's
    # response to the stimulus that was on screen at frame N (instead of
    # ~4.8 s later).
    if hrf_lag_trs > 0:
        for h in "LR":
            for d in (bold_mean, bold_t):
                shifted = np.zeros_like(d[h])
                shifted[:, :d[h].shape[1] - hrf_lag_trs] = d[h][:, hrf_lag_trs:]
                d[h] = shifted
    T = bold_mean["L"].shape[1]
    if max_frames is not None:
        T = min(T, max_frames)
        bold_mean = {h: bold_mean[h][:, :T] for h in "LR"}
        bold_t = {h: bold_t[h][:, :T] for h in "LR"}

    # Temporal upsampling: linear-interpolate bold/t between TRs so the
    # movie feels smooth instead of stepping once per TR.
    if temporal_upsample > 1:
        bold_mean = {h: _upsample_temporal(bold_mean[h], temporal_upsample)
                      for h in "LR"}
        bold_t = {h: _upsample_temporal(bold_t[h], temporal_upsample)
                   for h in "LR"}
    T_disp = bold_mean["L"].shape[1]

    print(f"[sub-{subject_id:02d}] loading inflated surfaces")
    surf_d = _load_inflated(subject)

    # Display BOLD: positive PSC only (negative is rare during sweep and
    # mostly noise; the t-stat mask doesn't distinguish sign by default).
    # Negative-PSC vertices end up at the cmap zero (transparent).
    bold_disp = {h: np.clip(bold_mean[h], 0.0, None).astype(np.float32)
                  for h in "LR"}
    if normalize == "per_vertex":
        cmap_vmin, cmap_vmax = 0.0, 1.0
        for h in "LR":
            peak = np.maximum(
                np.percentile(bold_disp[h], 95, axis=1, keepdims=True), 0.3
            )
            bold_disp[h] = bold_disp[h] / peak
    elif normalize == "absolute":
        cmap_vmin, cmap_vmax = 0.0, signal_clip
    else:
        raise ValueError(
            f"normalize must be 'absolute' or 'per_vertex', got {normalize!r}"
        )

    print(
        f"[sub-{subject_id:02d}] projecting hemispheres "
        f"({view} {lateral_angle_deg:.0f}°)"
    )
    renderers = {}
    extents = {}
    for h in "LR":
        coords3d = np.asarray(surf_d[h]["mesh"].coordinates, dtype=np.float32)
        faces = np.asarray(surf_d[h]["mesh"].faces, dtype=np.int64)
        coords_xy, faces_vis = _project_hemi(
            coords3d, faces, side=h, lateral_angle_deg=lateral_angle_deg,
            view=view,
        )
        vn = _compute_vertex_normals(coords3d, faces)
        # Headlamp light from the camera position, with a small upward
        # tilt so the tops of gyri are highlighted.
        theta = np.deg2rad(lateral_angle_deg)
        sign = -1.0 if h == "L" else 1.0
        if view == "medial":
            sign = -sign
        cam = np.array([sign * np.sin(theta), -np.cos(theta), 0.4])
        light = cam / np.linalg.norm(cam)
        renderer, extent = _make_brain_renderer(
            coords_xy, faces_vis, surf_d[h]["curv"],
            width=brain_width, height=brain_height,
            stat_vmin=cmap_vmin, stat_vmax=cmap_vmax,
            vertex_normals=vn, light_dir=light,
            cmap_name="magma",
            t_cutoff=t_cutoff,
        )
        renderers[h] = renderer
        extents[h] = extent

    print(f"[sub-{subject_id:02d}] computing bar trajectory (×{temporal_upsample})")
    bar_ori, bar_pos, bar_vis = _bar_state_per_TR(
        subject, sessions=(1,), runs=(1,), settings=settings,
        n_volumes=T, upsample=temporal_upsample,
    )

    with plt.rc_context(PRES_RC):
        panel_h = 6.0
        par_w = panel_h
        brain_w = panel_h
        fig_w = par_w + brain_w * 2 + 0.6
        fig_h = panel_h + 0.6
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[par_w, brain_w, brain_w],
            wspace=0.04, left=0.02, right=0.99,
            top=0.91, bottom=0.04,
        )
        ax_par = fig.add_subplot(gs[0, 0])
        ax_lh = fig.add_subplot(gs[0, 1])
        ax_rh = fig.add_subplot(gs[0, 2])

        bar_img, aperture, tiles = _paradigm_artists(
            ax_par, settings, half_extent=half_extent,
        )
        ax_par.set_title("Stimulus paradigm")

        im_lh = ax_lh.imshow(
            renderers["L"](bold_disp["L"][:, 0], bold_t["L"][:, 0]),
            extent=extents["L"], origin="lower", interpolation="bilinear",
        )
        im_rh = ax_rh.imshow(
            renderers["R"](bold_disp["R"][:, 0], bold_t["R"][:, 0]),
            extent=extents["R"], origin="lower", interpolation="bilinear",
        )
        for ax, ttl in ((ax_lh, "Left hemisphere"),
                         (ax_rh, "Right hemisphere")):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(ttl)

        signal_label = {"psc": "% signal change", "z": "z-score"}[signal]
        norm_label = ", per-vertex normalised" if normalize == "per_vertex" else ""
        view_label = "medial" if view == "medial" else "lateral"
        sup = fig.suptitle(
            f"Sub-{subject_id:02d} — cleaned BOLD ({signal_label}{norm_label}), "
            f"averaged over all runs  •  {view_label} view   TR 0",
        )

        TR = float(subject.get_tr(1, 1))
        TR_per_frame = 1.0 / temporal_upsample

        def update(t: int):
            _update_bar(bar_img, bar_ori[t], bar_pos[t],
                        bar_width=bar_w, rad=rad, visible=bool(bar_vis[t]),
                        tiles=tiles, frame_idx=t)
            im_lh.set_data(
                renderers["L"](bold_disp["L"][:, t], bold_t["L"][:, t])
            )
            im_rh.set_data(
                renderers["R"](bold_disp["R"][:, t], bold_t["R"][:, t])
            )
            tr_float = t * TR_per_frame
            sup.set_text(
                f"Sub-{subject_id:02d} — cleaned BOLD "
                f"({signal_label}{norm_label}), averaged over all runs  •  "
                f"{view_label} view   TR {tr_float:5.2f}  "
                f"({tr_float * TR:5.1f} s)"
            )
            return im_lh, im_rh, sup

        anim = manim.FuncAnimation(
            fig, update, frames=T_disp, interval=1000 / fps, blit=False,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".mp4":
            # h264/yuv420p needs even W and H. We pick a dpi that makes
            # figsize × dpi an even integer in both axes — no pad filter
            # (pad caused a shear in the rawvideo→yuv420p pipeline).
            target_w_px = int(round(fig_w * dpi))
            target_h_px = int(round(fig_h * dpi))
            # Bump dpi by 1 until both dims are even
            d = dpi
            while (int(round(fig_w * d)) % 2) or (int(round(fig_h * d)) % 2):
                d += 1
            writer = manim.FFMpegWriter(
                fps=fps, extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(out_path, writer=writer, dpi=d)
        elif suffix == ".gif":
            anim.save(out_path, writer="pillow", fps=fps,
                       dpi=max(70, dpi // 2))
        else:
            raise ValueError(f"Unsupported extension {suffix}")
        plt.close(fig)
    print(f"wrote: {out_path}  ({T_disp} frames @ {fps} fps  ≈ "
           f"{T_disp / fps:.1f} s)")


def _cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", type=int, default=[3])
    p.add_argument(
        "--bids-folder", type=Path, default=Path("/data/ds-retsupp"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/Users/gdehol/git/retsupp/notes/figures/talk"),
    )
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--model", type=int, default=1,
                    help="PRF model used for R² thresholding (default 1 = "
                         "Gaussian PRF with fixed HRF).")
    p.add_argument("--fdr-alpha", type=float, default=0.001,
                    help="BH-FDR q-value for masking vertices (default 1e-3).")
    p.add_argument("--signal", choices=("psc", "z"), default="psc",
                    help="Per-vertex normalisation. 'psc' = percent signal "
                         "change vs run-mean (physiological units); 'z' = "
                         "per-vertex z-score along time.")
    p.add_argument("--signal-clip", type=float, default=2.0,
                    help="Cmap upper clip in signal units (default 2.0%% "
                         "PSC). With --normalize per_vertex this is "
                         "ignored.")
    p.add_argument("--normalize", choices=("absolute", "per_vertex"),
                    default="absolute",
                    help="'absolute': cmap in raw signal units. "
                         "'per_vertex': divide each vertex by its 95th "
                         "percentile peak so vertex amplitudes are equalised.")
    p.add_argument("--views", nargs="+",
                    choices=("lateral", "medial"),
                    default=["lateral"],
                    help="One or more views to render. One movie per view "
                         "per subject.")
    p.add_argument("--t-cutoff", type=float, default=3.1,
                    help="Per-(vertex, TR) t-threshold for the time-"
                         "varying mask (signed: only positive activation "
                         "is shown). Default 3.1 ≈ uncorrected α=0.01 "
                         "two-sided with 12 runs (df=11).")
    p.add_argument("--temporal-upsample", type=int, default=4,
                    help="Linear-interpolate BOLD + bar trajectory to "
                         "this many frames per TR for smoother motion.")
    p.add_argument("--cache-only", action="store_true",
                    help="Just populate the vol-to-surf BOLD cache for each "
                         "subject and exit; don't render any movie. Use this "
                         "on the cluster to pre-compute caches in parallel.")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                    help=f"Where to read/write per-run vol-to-surf BOLD "
                         f"caches. Default: {CACHE_DIR}.")
    p.add_argument("--lateral-angle-deg", type=float, default=25.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--hrf-lag-trs", type=int, default=3,
                    help="Shift BOLD back by this many TRs so the brain "
                         "panel is roughly time-aligned with the stimulus "
                         "(default 3 ≈ 4.8 s HRF delay).")
    p.add_argument("--brain-width", type=int, default=520)
    p.add_argument("--brain-height", type=int, default=480)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--suffix", default=".mp4", choices=[".mp4", ".gif"])
    args = p.parse_args()

    if args.cache_only:
        for sid in args.subjects:
            print(f"\n=== caching surf BOLD for sub-{sid:02d} ===", flush=True)
            subject = Subject(sid, bids_folder=str(args.bids_folder))
            for ses in (1, 2):
                for run in subject.get_runs(session=ses):
                    print(f"  ses-{ses} run-{run}…", flush=True)
                    _cached_surf_bold(subject, ses, run,
                                       cache_dir=args.cache_dir)
            print(f"=== sub-{sid:02d} cached ===", flush=True)
        return

    for sid in args.subjects:
        for view in args.views:
            out = args.out_dir / (
                f"paradigm_brain_sub-{sid:02d}_avg-allruns_"
                f"{view}{args.suffix}"
            )
            render_subject(
                sid,
                bids_folder=args.bids_folder,
                out_path=out,
                fps=args.fps,
                model=args.model,
                fdr_alpha=args.fdr_alpha,
                signal=args.signal,
                signal_clip=args.signal_clip,
                normalize=args.normalize,
                view=view,
                t_cutoff=args.t_cutoff,
                lateral_angle_deg=args.lateral_angle_deg,
                brain_width=args.brain_width,
                brain_height=args.brain_height,
                max_frames=args.max_frames,
                hrf_lag_trs=args.hrf_lag_trs,
                temporal_upsample=args.temporal_upsample,
                dpi=args.dpi,
            )


if __name__ == "__main__":
    _cli()
