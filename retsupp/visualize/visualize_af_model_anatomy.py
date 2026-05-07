"""Didactic illustration of the joint AF+PRF model and its parameters.

A multi-page PDF that walks through what each shared parameter does
and what the model predicts. Useful for meetings/talks: "this is what
the model is and how σ_AF, g_HP, g_LP shape its predictions".

Pages:
  1. Cover with the formula and parameter glossary.
  2. The four ring attention fields A_ℓ(g) (one per ring location).
  3. Per-condition modulation field M_C(g) at typical (σ_AF, g_HP, g_LP).
  4. Effect of σ_AF — three columns at small / medium / large σ_AF.
  5. Effect of (g_HP, g_LP) — 3 × 3 grid sweeping each gain.
  6. Single-voxel example: S_v(g), M_C(g), R_v_C(g) = M_C·S_v, with
     arrow showing predicted center shift.
  7. Predicted shift vector field across visual space (one panel per
     HP condition — the same idea as visualize_af_prediction).

Usage
-----
    python -m retsupp.visualize.visualize_af_model_anatomy \\
        --out notes/af_model_anatomy.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

from retsupp.utils.data import distractor_locations


CONDITIONS = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
RING_KEYS = ['upper right', 'upper left', 'lower left', 'lower right']
APERTURE = 3.17  # bar aperture radius (deg)
RING_R = 4.0     # eccentricity of the 4 ring positions


def get_ring_positions():
    return np.array([list(distractor_locations[k]) for k in RING_KEYS],
                    dtype=np.float32)


def make_grid(resolution=121, grid_radius=5.0):
    g1d = np.linspace(-grid_radius, grid_radius, resolution).astype(np.float32)
    GX, GY = np.meshgrid(g1d, g1d)
    return GX, GY, g1d


def ring_gaussian(GX, GY, mu, sigma):
    return np.exp(-((GX - mu[0])**2 + (GY - mu[1])**2) / (2 * sigma**2))


def modulation_field(GX, GY, ring, hp_idx, sigma_AF, g_HP, g_LP, mode='signed'):
    sign = +1.0 if mode != 'suppression' else -1.0
    if mode in ('attraction', 'signed'):
        sign = +1.0
    elif mode == 'suppression':
        sign = -1.0
    A = np.stack([ring_gaussian(GX, GY, ring[i], sigma_AF)
                   for i in range(len(ring))], axis=0)   # (4, R, R)
    a_hp = A[hp_idx]
    a_lp = A.sum(axis=0) - a_hp
    M = 1.0 + sign * (g_HP * a_hp + g_LP * a_lp)
    return np.maximum(M, 0.0)


def add_aperture_and_ring(ax, ring, hp_idx=None):
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='0.5', ls='--', lw=0.6))
    for i in range(len(ring)):
        c = 'C3' if (hp_idx is not None and i == hp_idx) else 'gray'
        ax.plot(ring[i, 0], ring[i, 1], 'o', markersize=8,
                color=c, mec='k', alpha=0.85)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def cover_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111); ax.axis('off')
    ax.text(0.5, 0.95, 'Joint Attention-Field × PRF model',
            ha='center', va='top', fontsize=20, weight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.91, 'anatomy of the parameters',
            ha='center', va='top', fontsize=12, style='italic',
            transform=ax.transAxes)

    formula = (
        '\nForward model (per voxel v, per HP condition C):\n\n'
        '   R_v_C(g) =  M_C(g)  ·  S_v(g)\n\n'
        '   M_C(g)  =  1  +  sign · ( g_HP · A_{H_C}(g)  +  g_LP · Σ_{ℓ ≠ H_C} A_ℓ(g) )\n\n'
        '   A_ℓ(g)  =  exp( -‖g − μ_ℓ‖² / (2 σ_AF²) )         [ unit peak ]\n\n'
        '   S_v(g)  =  Gaussian PRF at (x_v, y_v) with width sd_v\n'
    )
    glossary = (
        '\nShared parameters (same value for every voxel in a ROI):\n'
        '   σ_AF       width of each AF Gaussian — how spatially focused\n'
        '              the modulation is around each ring location.\n'
        '   g_HP       modulation strength AT THE HP LOCATION\n'
        '              ( negative = suppression,  positive = attraction )\n'
        '   g_LP       modulation strength AT EACH OF THE 3 LP LOCATIONS\n'
        '              (same convention as g_HP).\n\n'
        'Per-voxel parameters: x, y, sd, baseline, amplitude (Gaussian PRF).\n\n'
        'Quantity of interest:  g_HP − g_LP  or  log( (1+g_HP)/(1+g_LP) )\n'
        '   negative  ⇒  HP-specific suppression (vs surrounding LP).\n'
    )
    ax.text(0.04, 0.86, formula, ha='left', va='top',
            family='monospace', fontsize=10, transform=ax.transAxes)
    ax.text(0.04, 0.55, glossary, ha='left', va='top',
            family='monospace', fontsize=9, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def page_ring_AFs(pdf, sigma_AF=0.8):
    GX, GY, _ = make_grid()
    ring = get_ring_positions()

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(
        f'(2/7)  The four ring attention fields  A_ℓ(g)   '
        f'( σ_AF = {sigma_AF}° )',
        fontsize=12,
    )
    # 1×4 individual AFs.
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 1)
        A = ring_gaussian(GX, GY, ring[i], sigma_AF)
        ax.imshow(A, extent=(-5, 5, -5, 5), origin='lower',
                  cmap='Reds', vmin=0, vmax=1)
        add_aperture_and_ring(ax, ring, hp_idx=i)
        ax.set_title(f'A_{CONDITIONS[i]}', fontsize=9)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # Sum of all four.
    ax = fig.add_subplot(2, 4, 5)
    A_sum = np.stack([ring_gaussian(GX, GY, ring[i], sigma_AF)
                       for i in range(4)], axis=0).sum(axis=0)
    ax.imshow(A_sum, extent=(-5, 5, -5, 5), origin='lower',
              cmap='Reds', vmin=0, vmax=1)
    add_aperture_and_ring(ax, ring)
    ax.set_title('Σ_ℓ A_ℓ', fontsize=9)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # M for HP=upper_right with typical gains.
    for col, (g_HP, g_LP) in enumerate([(0.5, 0.2), (-0.5, -0.2),
                                          (0.5, -0.2)]):
        ax = fig.add_subplot(2, 4, 6 + col)
        M = modulation_field(GX, GY, ring, hp_idx=0,
                              sigma_AF=sigma_AF,
                              g_HP=g_HP, g_LP=g_LP, mode='signed')
        vmax = max(1.0, M.max())
        ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                  cmap='RdBu_r', vmin=2 - vmax, vmax=vmax)
        add_aperture_and_ring(ax, ring, hp_idx=0)
        ax.set_title(f'M_UR(g)  g_HP={g_HP}, g_LP={g_LP}',
                     fontsize=8)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def page_modulation_for_4_conditions(pdf, sigma_AF=0.8, g_HP=-0.5, g_LP=-0.1):
    GX, GY, _ = make_grid()
    ring = get_ring_positions()

    fig = plt.figure(figsize=(13, 4.5))
    fig.suptitle(
        f'(3/7)  M_C(g) for each HP condition  '
        f'(σ_AF={sigma_AF}, g_HP={g_HP}, g_LP={g_LP}; '
        f'this example: HP suppressed)',
        fontsize=11,
    )
    for ci, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(1, 4, ci + 1)
        M = modulation_field(GX, GY, ring, hp_idx=ci, sigma_AF=sigma_AF,
                              g_HP=g_HP, g_LP=g_LP, mode='signed')
        vmax = max(M.max(), 2.0 - M.min())
        ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                  cmap='RdBu_r', vmin=2 - vmax, vmax=vmax)
        add_aperture_and_ring(ax, ring, hp_idx=ci)
        ax.set_title(f'HP = {cond}', fontsize=9)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_sigma_sweep(pdf, sigmas=(0.5, 1.5, 3.0), g_HP=-0.5, g_LP=-0.1):
    GX, GY, _ = make_grid()
    ring = get_ring_positions()

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        f'(4/7)  σ_AF effect — same gains (g_HP={g_HP}, g_LP={g_LP}), HP=UR.\n'
        'Smaller σ_AF → spatially focused modulation; '
        'larger σ_AF → diffuse / overlapping fields.',
        fontsize=11,
    )
    for i, sigma in enumerate(sigmas):
        ax = fig.add_subplot(1, len(sigmas), i + 1)
        M = modulation_field(GX, GY, ring, hp_idx=0, sigma_AF=sigma,
                              g_HP=g_HP, g_LP=g_LP, mode='signed')
        vmax = max(M.max(), 2.0 - M.min())
        ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                  cmap='RdBu_r', vmin=2 - vmax, vmax=vmax)
        add_aperture_and_ring(ax, ring, hp_idx=0)
        ax.set_title(f'σ_AF = {sigma}°', fontsize=10)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig); plt.close(fig)


def page_gain_sweep(pdf, sigma_AF=1.0,
                      g_HPs=(-0.6, 0.0, +0.6),
                      g_LPs=(-0.3, 0.0, +0.3)):
    GX, GY, _ = make_grid()
    ring = get_ring_positions()

    fig = plt.figure(figsize=(11, 11))
    fig.suptitle(
        f'(5/7)  Gain sweep — σ_AF = {sigma_AF}°, HP = UR.\n'
        'Rows: g_HP (HP-location strength).  '
        'Columns: g_LP (LP-location strength).\n'
        'Negative ⇒ suppression (blue), positive ⇒ attraction (red).',
        fontsize=11,
    )
    for i, g_HP in enumerate(g_HPs):
        for j, g_LP in enumerate(g_LPs):
            ax = fig.add_subplot(len(g_HPs), len(g_LPs),
                                  i * len(g_LPs) + j + 1)
            M = modulation_field(GX, GY, ring, hp_idx=0, sigma_AF=sigma_AF,
                                  g_HP=g_HP, g_LP=g_LP, mode='signed')
            ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                      cmap='RdBu_r', vmin=0.4, vmax=1.6)
            add_aperture_and_ring(ax, ring, hp_idx=0)
            ax.set_title(f'g_HP={g_HP:+.1f}, g_LP={g_LP:+.1f}', fontsize=8)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_voxel_example(pdf, voxel_xy=(0.0, 1.5), prf_sd=1.0,
                        sigma_AF=1.0, g_HP=-0.5, g_LP=-0.1, hp_idx=0):
    GX, GY, _ = make_grid()
    ring = get_ring_positions()
    M = modulation_field(GX, GY, ring, hp_idx=hp_idx, sigma_AF=sigma_AF,
                          g_HP=g_HP, g_LP=g_LP, mode='signed')
    S = ring_gaussian(GX, GY, voxel_xy, prf_sd)
    R = M * S
    R_pos = np.clip(R, 0, None)
    Z = R_pos.sum() + 1e-12
    pred_x = (GX * R_pos).sum() / Z
    pred_y = (GY * R_pos).sum() / Z

    fig = plt.figure(figsize=(13, 4.5))
    fig.suptitle(
        f'(6/7)  Single-voxel example.  '
        f'PRF at ({voxel_xy[0]:+.1f}, {voxel_xy[1]:+.1f}), sd={prf_sd}.  '
        f'HP = {CONDITIONS[hp_idx]}.  '
        f'σ_AF={sigma_AF}, g_HP={g_HP}, g_LP={g_LP}.',
        fontsize=11,
    )
    panels = [
        ('S_v(g)  — voxel PRF', S, 'Greys'),
        ('M_C(g)  — modulation', M, 'RdBu_r'),
        ('R_v_C(g) = M_C · S_v', R, 'Greys'),
    ]
    for i, (title, arr, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1)
        if cmap == 'RdBu_r':
            ax.imshow(arr, extent=(-5, 5, -5, 5), origin='lower',
                      cmap=cmap, vmin=0.4, vmax=1.6)
        else:
            ax.imshow(arr, extent=(-5, 5, -5, 5), origin='lower',
                      cmap=cmap, vmin=0)
        add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
        ax.plot(voxel_xy[0], voxel_xy[1], 'gx', markersize=10,
                markeredgewidth=2, label='base PRF center')
        if i == 2:
            ax.plot(pred_x, pred_y, 'g+', markersize=14,
                    markeredgewidth=2, label='predicted AD-pRF center')
            ax.annotate('', xy=(pred_x, pred_y),
                        xytext=(voxel_xy[0], voxel_xy[1]),
                        arrowprops=dict(arrowstyle='->',
                                          color='C2', lw=2))
            ax.legend(fontsize=7, loc='upper left')
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


def predict_shift_field(sigma_AF, g_HP, g_LP,
                         seeds, integration_resolution=81,
                         integration_radius=5.0, prf_sd=1.0, mode='signed'):
    GX, GY, _ = make_grid(resolution=integration_resolution,
                            grid_radius=integration_radius)
    ring = get_ring_positions()
    pred = np.empty((len(CONDITIONS), seeds.shape[0], 2), dtype=np.float32)
    for ci in range(len(CONDITIONS)):
        M = modulation_field(GX, GY, ring, hp_idx=ci, sigma_AF=sigma_AF,
                              g_HP=g_HP, g_LP=g_LP, mode=mode)
        for vi, (vx, vy) in enumerate(seeds):
            S = np.exp(-((GX - vx)**2 + (GY - vy)**2) / (2 * prf_sd**2))
            rho = np.clip(M * S, 0, None)
            Z = rho.sum() + 1e-12
            pred[ci, vi, 0] = (GX * rho).sum() / Z
            pred[ci, vi, 1] = (GY * rho).sum() / Z
    return pred


def page_vector_field(pdf, sigma_AF=1.0, g_HP=-0.5, g_LP=-0.1,
                        prf_sd=1.0, spacing=0.6, fov=4.0, scale=5.0,
                        aperture_only=False):
    s1d = np.arange(-fov, fov + spacing / 2, spacing)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    if aperture_only:
        keep = np.linalg.norm(seeds, axis=1) <= APERTURE
        seeds = seeds[keep]
    pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds, prf_sd=prf_sd)
    ring = get_ring_positions()

    fig = plt.figure(figsize=(11, 11))
    label = 'inside aperture only' if aperture_only else 'full FOV'
    fig.suptitle(
        f'(7/7)  Predicted PRF-center shift field  ({label})  '
        f'(σ_AF={sigma_AF}, g_HP={g_HP}, g_LP={g_LP}, prf_sd={prf_sd})\n'
        f'Each arrow goes from a seed PRF center to the predicted AD-pRF '
        f'center (arrows scaled ×{scale}).',
        fontsize=11,
    )
    for ci, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(2, 2, ci + 1)
        dx = pred[ci, :, 0] - seeds[:, 0]
        dy = pred[ci, :, 1] - seeds[:, 1]
        add_aperture_and_ring(ax, ring, hp_idx=ci)
        ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
                  angles='xy', scale_units='xy',
                  scale=1.0 / scale, color='C0', width=0.003, alpha=0.85)
        ax.scatter(seeds[:, 0], seeds[:, 1], s=1, color='k', alpha=0.4)
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f'HP = {cond}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_projection_field(pdf, sigma_AF=2.0, g_HP=+0.14, g_LP=+0.29,
                            prf_sd=1.0, spacing=0.4, fov=4.5,
                            integration_resolution=81,
                            integration_radius=5.0,
                            aperture_only=False):
    """Why the away-from-HP projection is small at large distance from HP.

    For each seed PRF position the model predicts a 2D shift vector
    (pred − base). We project that onto the unit vector pointing AWAY
    from HP, exactly as the data analysis does. The resulting scalar
    field shows why the projection is large near HP, small at large
    distance, and ~0 at the perpendicular-to-HP locations: shifts there
    are radial wrt the nearby LP-AF, hence tangential to the HP-axis,
    so the projection cancels.
    """
    s1d = np.arange(-fov, fov + spacing / 2, spacing)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    pred = predict_shift_field(sigma_AF, g_HP, g_LP, seeds,
                                 integration_resolution=integration_resolution,
                                 integration_radius=integration_radius,
                                 prf_sd=prf_sd)
    inside_mask = np.linalg.norm(seeds, axis=1) <= APERTURE
    ring = get_ring_positions()
    hp_idx = 0  # UR
    hp = ring[hp_idx]
    u = -hp / (np.linalg.norm(hp) + 1e-12)   # away-from-HP unit vector

    dx = pred[hp_idx, :, 0] - seeds[:, 0]
    dy = pred[hp_idx, :, 1] - seeds[:, 1]
    proj = dx * u[0] + dy * u[1]              # away-from-HP projection
    mag = np.sqrt(dx ** 2 + dy ** 2)
    if aperture_only:
        proj = np.where(inside_mask, proj, np.nan)
        mag = np.where(inside_mask, mag, np.nan)

    # Reshape onto the seed grid for imshow.
    n_side = SX.shape[0]
    proj_grid = proj.reshape(n_side, n_side)
    mag_grid = mag.reshape(n_side, n_side)

    fig = plt.figure(figsize=(13, 5.5))
    fig.suptitle(
        'Why projection on the away-from-HP axis is small at large distance.'
        + ('  (Aperture-only — voxels with |position| > 3.17° masked.)'
           if aperture_only else '') +
        f'\nσ_AF={sigma_AF}, g_HP={g_HP:+.2f}, g_LP={g_LP:+.2f}, '
        f'prf_sd={prf_sd}.  HP = upper-right (red star).',
        fontsize=11,
    )

    extent = (s1d[0], s1d[-1], s1d[0], s1d[-1])

    # (1) shift-magnitude scalar field — full magnitude, regardless of direction.
    ax = fig.add_subplot(1, 3, 1)
    vmax_mag = float(np.nanpercentile(mag_grid, 98)) or 0.01
    im = ax.imshow(mag_grid, extent=extent, origin='lower',
                    cmap='Greys', vmin=0, vmax=vmax_mag)
    add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
    ax.plot(hp[0], hp[1], '*', markersize=18, color='C3', mec='k', zorder=5)
    ax.set_xlim(-fov, fov); ax.set_ylim(-fov, fov)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('|shift|  (any direction)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label='deg')

    # (2) projection onto away-from-HP axis (signed scalar field).
    ax = fig.add_subplot(1, 3, 2)
    vmax_proj = float(np.nanpercentile(np.abs(proj_grid), 98)) or 0.01
    im = ax.imshow(proj_grid, extent=extent, origin='lower',
                    cmap='RdBu_r', vmin=-vmax_proj, vmax=+vmax_proj)
    add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
    ax.plot(hp[0], hp[1], '*', markersize=18, color='k', mec='w', zorder=5)
    # The HP-axis line.
    ax.plot([-fov * u[0], fov * u[0]],
            [-fov * u[1], fov * u[1]],
            '--', color='gray', lw=0.7)
    ax.set_xlim(-fov, fov); ax.set_ylim(-fov, fov)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('projection onto away-from-HP axis\n'
                 '(signed: red = away, blue = toward)',
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label='deg')

    # (3) shift VECTORS (so direction is explicit).
    ax = fig.add_subplot(1, 3, 3)
    if aperture_only:
        ix = inside_mask
        ax.quiver(seeds[ix, 0], seeds[ix, 1], dx[ix], dy[ix],
                  angles='xy', scale_units='xy',
                  scale=1.0 / 5.0, color='C0',
                  width=0.005, alpha=0.85)
    else:
        ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
                  angles='xy', scale_units='xy',
                  scale=1.0 / 5.0, color='C0',
                  width=0.005, alpha=0.85)
    add_aperture_and_ring(ax, ring, hp_idx=hp_idx)
    ax.plot(hp[0], hp[1], '*', markersize=18, color='C3', mec='k', zorder=5)
    ax.set_xlim(-fov, fov); ax.set_ylim(-fov, fov)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('predicted shift vectors (×5)', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)

    # Bonus page: 1D azimuthal cut along the HP↔origin↔opposite-LP line.
    # Walk a probe point from HP outward through origin to the opposite
    # ring location and show |shift| and projection vs distance from HP.
    n_pts = 80
    distances = np.linspace(-1.0, 8.0, n_pts)   # from HP outward
    base_pts = hp[None, :] + distances[:, None] * u[None, :]
    pred_pts = predict_shift_field(sigma_AF, g_HP, g_LP,
                                     base_pts.astype(np.float32),
                                     integration_resolution=integration_resolution,
                                     integration_radius=integration_radius,
                                     prf_sd=prf_sd)
    dx_pts = pred_pts[hp_idx, :, 0] - base_pts[:, 0]
    dy_pts = pred_pts[hp_idx, :, 1] - base_pts[:, 1]
    proj_pts = dx_pts * u[0] + dy_pts * u[1]
    mag_pts = np.sqrt(dx_pts ** 2 + dy_pts ** 2)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(distances, mag_pts, 'k-', lw=2, label='|shift| (any direction)')
    ax.plot(distances, proj_pts, 'C3-', lw=2,
            label='projection on away-from-HP axis')
    ax.fill_between(distances, 0, mag_pts, color='0.85', zorder=0)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='C3', lw=0.7, ls=':', label='HP location')
    # Label LL position approximately (twice the HP magnitude).
    d_ll = 2.0 * np.linalg.norm(hp)
    ax.axvline(d_ll, color='gray', lw=0.7, ls=':', label='opposite LP (LL)')
    ax.set_xlabel('distance along HP→opposite axis  (deg, from HP)')
    ax.set_ylabel('shift size  (deg)')
    ax.set_title(
        'Predicted shift along the HP-axis line (HP at d=0, opposite LP at d≈5.7°).\n'
        'Total |shift| can be non-zero at far distance, but its projection on\n'
        'the HP-axis stays small — shifts there are radial wrt the LL-AF, \n'
        'NOT wrt HP, so they cancel along this projection.',
        fontsize=10,
    )
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_per_roi_typical(pdf, params_per_roi):
    """Show each ROI's modulation field at its empirical median params.

    `params_per_roi` is a list of dicts with keys: roi, sigma_AF, g_HP,
    g_LP, n. Empirical values pulled from notes/af_parameters.tsv.
    """
    GX, GY, _ = make_grid()
    ring = get_ring_positions()
    n = len(params_per_roi)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow + 1.2))
    fig.suptitle(
        'Per-ROI modulation field at the empirical median parameters '
        '(across all subjects).\n'
        'HP shown at upper-right (rotation symmetry — only sign of '
        'g_HP − g_LP matters).',
        fontsize=11,
    )
    for i, p in enumerate(params_per_roi):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        M = modulation_field(GX, GY, ring, hp_idx=0,
                              sigma_AF=p['sigma_AF'],
                              g_HP=p['g_HP'], g_LP=p['g_LP'], mode='signed')
        # Symmetric colour scale around 1.
        d = max(abs(M.max() - 1.0), abs(1.0 - M.min()), 0.05)
        ax.imshow(M, extent=(-5, 5, -5, 5), origin='lower',
                  cmap='RdBu_r', vmin=1 - d, vmax=1 + d)
        add_aperture_and_ring(ax, ring, hp_idx=0)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{p['roi']}  (n={p['n']})\n"
            f"σ={p['sigma_AF']:.2f}, g_HP={p['g_HP']:+.2f}, "
            f"g_LP={p['g_LP']:+.2f}",
            fontsize=8,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def page_per_roi_vector_field(pdf, params_per_roi, prf_sd=1.0,
                                spacing=0.7, fov=4.0, scale=10.0,
                                aperture_only=False):
    """Per-ROI vector field at empirical median params (HP=upper_right)."""
    s1d = np.arange(-fov, fov + spacing / 2, spacing)
    SX, SY = np.meshgrid(s1d, s1d)
    seeds = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    if aperture_only:
        seeds = seeds[np.linalg.norm(seeds, axis=1) <= APERTURE]
    ring = get_ring_positions()

    n = len(params_per_roi)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow + 1.2))
    fig.suptitle(
        f'Per-ROI predicted shift vector field at empirical median '
        f'parameters (HP = upper_right; arrows ×{scale} for visibility)',
        fontsize=11,
    )
    for i, p in enumerate(params_per_roi):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        pred = predict_shift_field(p['sigma_AF'], p['g_HP'], p['g_LP'],
                                     seeds, prf_sd=prf_sd)
        dx = pred[0, :, 0] - seeds[:, 0]
        dy = pred[0, :, 1] - seeds[:, 1]
        add_aperture_and_ring(ax, ring, hp_idx=0)
        ax.quiver(seeds[:, 0], seeds[:, 1], dx, dy,
                  angles='xy', scale_units='xy',
                  scale=1.0 / scale, color='C0',
                  width=0.005, alpha=0.85)
        ax.scatter(seeds[:, 0], seeds[:, 1], s=1, color='k', alpha=0.4)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{p['roi']}  (n={p['n']})\n"
            f"σ={p['sigma_AF']:.2f}, g_diff={p['g_HP']-p['g_LP']:+.2f}",
            fontsize=8,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def load_typical_params(tsv_path: Path):
    """Per-ROI median parameters from a plot_af_parameters.py TSV dump."""
    if not tsv_path.exists():
        return []
    df = pd.read_csv(tsv_path, sep='\t')
    out = []
    # Preserve the canonical hierarchy order.
    canonical_order = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']
    for roi in canonical_order:
        sub = df[df['roi'] == roi]
        if len(sub) == 0:
            continue
        out.append(dict(
            roi=roi, n=len(sub),
            sigma_AF=float(sub['sigma_AF'].median()),
            g_HP=float(sub['g_HP'].median()),
            g_LP=float(sub['g_LP'].median()),
        ))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=Path('notes/af_model_anatomy.pdf'))
    # Defaults match the empirical V3 median (where the HP-suppression
    # signature is strongest in our data — see plot_af_parameters.py).
    parser.add_argument('--sigma-af', type=float, default=1.92,
                        help='σ_AF (default: V3 median across subjects).')
    parser.add_argument('--g-hp',     type=float, default=+0.14,
                        help='g_HP (default: V3 median).')
    parser.add_argument('--g-lp',     type=float, default=+0.29,
                        help='g_LP (default: V3 median; g_HP < g_LP ⇒ '
                             'HP-specific suppression).')
    parser.add_argument('--params-tsv', type=Path,
                        default=Path('notes/af_parameters.tsv'),
                        help='Optional: per-fit parameter dump from '
                             'plot_af_parameters.py for the '
                             'per-ROI typical-values pages.')
    args = parser.parse_args()

    typical = load_typical_params(args.params_tsv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        cover_page(pdf)
        page_ring_AFs(pdf, sigma_AF=args.sigma_af)
        page_modulation_for_4_conditions(pdf,
                                            sigma_AF=args.sigma_af,
                                            g_HP=args.g_hp,
                                            g_LP=args.g_lp)
        page_sigma_sweep(pdf, g_HP=args.g_hp, g_LP=args.g_lp)
        page_gain_sweep(pdf, sigma_AF=args.sigma_af)
        page_voxel_example(pdf,
                            sigma_AF=args.sigma_af,
                            g_HP=args.g_hp, g_LP=args.g_lp)
        page_vector_field(pdf,
                            sigma_AF=args.sigma_af,
                            g_HP=args.g_hp, g_LP=args.g_lp)
        page_projection_field(pdf,
                                sigma_AF=args.sigma_af,
                                g_HP=args.g_hp, g_LP=args.g_lp)
        # Same two pages, restricted to seeds INSIDE the bar aperture
        # (radius 3.17°) — i.e. the visual-field region the experiment
        # actually probes with stimuli.
        page_vector_field(pdf,
                            sigma_AF=args.sigma_af,
                            g_HP=args.g_hp, g_LP=args.g_lp,
                            aperture_only=True)
        page_projection_field(pdf,
                                sigma_AF=args.sigma_af,
                                g_HP=args.g_hp, g_LP=args.g_lp,
                                aperture_only=True)
        if typical:
            page_per_roi_typical(pdf, typical)
            page_per_roi_vector_field(pdf, typical)
            page_per_roi_vector_field(pdf, typical, aperture_only=True)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
