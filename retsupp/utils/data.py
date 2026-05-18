import importlib.resources
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from nilearn import image, input_data, surface
import nibabel as nib
from itertools import product

# Utility function to load subject IDs from the installed YAML file
def get_subject_ids():
    """Load subject IDs from the installed subjects.yml and return as zero-padded strings."""
    import yaml
    with importlib.resources.files('retsupp.data').joinpath('subjects.yml').open('r') as f:
        subjects = yaml.safe_load(f)
    # Assume subjects is a list of ints or strings convertible to int
    return [f"{int(s):02d}" for s in subjects]

def get_retinotopic_labels():
    return {
        1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO1', 6: 'VO2', 7: 'LO1', 8: 'LO2',
        9: 'TO1', 10: 'TO2', 11: 'V3A', 12: 'V3B',}

roi_order = ['V1', 'V2', 'V3', 'V3AB', 'hV4', 'LO', 'TO', 'VO']


# Generic post-fit helpers live in braincoder.utils.postfit; re-export
# here so retsupp code that still imports them from this module
# (including in-flight cluster jobs) keeps working. Optional so that
# braincoder-less envs (e.g. pycortex2 used by visualize/) can still
# import Subject from this module.
try:
    from braincoder.utils.postfit import (  # noqa: E402, F401
        mask_valid_bold_voxels, mark_invalid_fits, validate_prf_parameters,
    )
except ImportError:
    pass


def select_well_fit_voxels(df, *, r2_threshold,
                            mass_threshold=0.5,
                            sigma_floor=0.30, sigma_ceil=4.0,
                            aperture_radius=3.17):
    """Apply the canonical voxel-selection filter for PRF analyses.

    Pure-filter — does not compute any threshold itself. Callers
    supply ``r2_threshold`` (typically via
    :meth:`Subject.get_r2_fdr_threshold`, which gives the cached
    logit-Gaussian tail-FDR threshold for that (subject, model, ROI)).

    Filters:
        - ``r² ≥ r2_threshold`` AND the row is "real" (r² > 0 and σ not
          NaN — the ``mark_invalid_fits`` sentinels are excluded).
        - ≥ ``mass_threshold`` of the PRF mass inside the bar
          aperture (Gaussian radial-CDF approximation, same as
          ``retsupp.visualize.utils.filter_prf_inside_aperture``).
        - σ ∈ [``sigma_floor``, ``sigma_ceil``] — drops the
          pathologically small / large fits.

    Args:
        df: per-voxel parameter DataFrame; must have ``r2``, ``sd``,
            ``x``, ``y`` columns. If ``eccen`` and ``mass_in`` are
            missing they are computed in a returned copy.
        r2_threshold: R² cutoff (typically from the cached logit-Gaussian
            tail-FDR helper). Use ``np.inf`` to disable the R² filter
            entirely (will pass everything else through).
        mass_threshold: minimum fraction of PRF mass that must lie
            inside the aperture.
        sigma_floor, sigma_ceil: σ bounds (degrees).
        aperture_radius: bar aperture radius in degrees (retsupp
            default 3.17 = 4 − 1.5/1.8).

    Returns:
        ``(selected_df, r2_threshold)`` — the rows of ``df`` that pass
        all filters (with ``eccen`` and ``mass_in`` columns guaranteed
        present), and the R² threshold passed in (echoed for caller
        convenience / legacy 2-tuple unpacking).
    """
    import numpy as np
    from scipy.stats import norm

    if not {"r2", "sd", "x", "y"}.issubset(df.columns):
        raise ValueError(
            "select_well_fit_voxels requires r2, sd, x, y columns")

    out = df.copy()
    if "eccen" not in out.columns:
        out["eccen"] = np.sqrt(out["x"] ** 2 + out["y"] ** 2)
    if "mass_in" not in out.columns:
        sd_safe = out["sd"].clip(lower=0.05)
        out["mass_in"] = 1.0 - norm.cdf(
            (out["eccen"] - aperture_radius) / sd_safe)

    real = (out["r2"] > 0) & out["sd"].notna()
    sel = (real
           & (out["r2"] >= r2_threshold)
           & (out["mass_in"] >= mass_threshold)
           & (out["sd"] >= sigma_floor)
           & (out["sd"] <= sigma_ceil))
    return out[sel].copy(), r2_threshold


class Subject(object):

    def __init__(self, subject_id, bids_folder='/data/ds-retsupp'):
        self.subject_id = int(subject_id)
        self.bids_folder = Path(bids_folder)
        # Mapping from distractor code to location
        self.location_mapping = {
            1.0: 'upper_right',
            3.0: 'upper_left',
            5.0: 'lower_left',
            7.0: 'lower_right',
            10.0: 'no distractor',
            np.nan: 'no distractor'
        }

    def get_hpd_locations(self):

        hpd_locations = {}

        for session in [1,2]:
            runs = self.get_runs(session)
            for run in runs:
                onsets = self.get_onsets(session=session, run=run)
                loc_counts = onsets[onsets.event_type == 'feedback']['distractor_location'].value_counts()
                loc_counts.index = loc_counts.index.map(self.location_mapping)

                hpd_locations[(session, run)] = loc_counts.idxmax()

        return hpd_locations

    def get_runs_by_hp(self, hp=None):
        """Group (session, run) pairs by their HP-distractor condition.

        Parameters
        ----------
        hp : str or None
            If None, returns the full mapping ``{hp_label:
            [(session, run), ...]}`` keyed by the 4 HP strings
            ('upper_right', 'upper_left', 'lower_left', 'lower_right').
            If a specific HP string, returns just that condition's
            run list.

        Returns
        -------
        dict[str, list[tuple[int, int]]] or list[tuple[int, int]]
            Per-HP run lists, or a single condition's run list.
        """
        groups: dict[str, list[tuple[int, int]]] = {}
        for sr, h in self.get_hpd_locations().items():
            groups.setdefault(h, []).append(sr)
        for k in groups:
            groups[k].sort()
        if hp is None:
            return groups
        if hp not in groups:
            raise KeyError(
                f'HP {hp!r} not found for sub-{self.subject_id:02d}. '
                f'Available: {sorted(groups)}')
        return groups[hp]

    def get_bar_stimulus(self, session=1, run=1, resolution=50,
                         grid_radius=5.0):
        """Bar-only PRF stimulus on the ±``grid_radius``° extended grid.

        Same bar geometry as :meth:`get_stimulus` and the bar pass of
        :meth:`get_stimulus_with_distractors`, but rendered on the
        wider grid (default 5°) instead of the bar aperture's 3.17°
        so it can be paired with PRF fits / decodes that use the
        extended grid coordinates.
        """
        settings = self.get_experimental_settings(session, run)
        tr = self.get_tr(session, run)
        n_volumes = self.get_n_volumes(session, run)
        frametimes = np.arange(tr / 2., tr * n_volumes + tr / 2., tr)

        bar_aperture = settings['radius_bar_aperture']
        bar_width = settings['bar_width']
        speed = settings['speed']
        fov_size = settings['fov_size']

        gx, gy = self.get_extended_grid_coordinates(
            resolution=resolution, session=session, run=run,
            grid_radius=grid_radius)
        aperture = np.sqrt(gx ** 2 + gy ** 2) <= bar_aperture

        onsets = self.get_onsets(session, run)
        bar_events = onsets[onsets['event_type'].apply(
            lambda s: s.startswith('bar'))]

        stim = np.zeros((len(frametimes), resolution, resolution),
                         dtype=np.float32)
        ori, pos = 0, -fov_size - bar_width

        for i, t in enumerate(frametimes):
            if t < bar_events['onset'].min():
                continue
            state_row = bar_events[bar_events['onset'] < t].iloc[-1]
            state = state_row['event_type']
            dt = t - state_row['onset']

            if state in ('bar_rest', 'bar_break'):
                continue
            if state == 'bar_right':
                ori, pos = 0, -bar_aperture - bar_width / 2 + dt * speed
            elif state == 'bar_left':
                ori, pos = 0, bar_aperture + bar_width / 2 - dt * speed
            elif state == 'bar_up':
                ori, pos = 90, -bar_aperture - bar_width / 2 + dt * speed
            elif state == 'bar_down':
                ori, pos = 90, bar_aperture + bar_width / 2 - dt * speed
            else:
                continue

            frame = np.zeros_like(gx, dtype=np.float32)
            if ori == 0:
                frame[np.abs(gx - pos) < bar_width / 2] = 1.0
            else:
                frame[np.abs(gy - pos) < bar_width / 2] = 1.0
            stim[i] = frame * aperture
        return stim

    def get_run_position_per_tr(self, session, run, hp_for_runs=None):
        """Position of (session, run) within its HP-condition block.

        Each subject's 12 (or fewer) runs are grouped chronologically by
        the run-level high-probability distractor (HP) condition. Within
        each group of runs sharing the same HP, this method returns the
        chronological index (0, 1, 2, ...) of the given (session, run).

        Examples
        --------
        - For a typical subject with order ``AAA BBB CCC DDD`` (3 runs
          per HP), the runs at ``(ses=1, run=1..3)`` map to positions
          ``0, 1, 2`` for HP=A; ``(ses=1, run=4..6)`` map to ``0, 1, 2``
          for HP=B; etc.
        - For sub-1/sub-2 with the bugged order ``AA-BBB-CCC-DDD-A``,
          the A-condition runs at ``(1,1), (1,2), (2,6)`` map to
          positions ``0, 1, 2`` (chronologically).
        - For sub-20 ses-1 (5 runs) and sub-24 ses-2 (5 runs) the last
          HP block is truncated to 2 runs -> positions ``0, 1``.

        Parameters
        ----------
        session : int
            Session index (1 or 2).
        run : int
            Run index within the session.
        hp_for_runs : dict[(int, int), str] or None
            Mapping ``(session, run) -> hp_string``. If None, the
            mapping is recomputed via :meth:`get_hpd_locations` (slow
            because it re-reads the events.tsv files; pass it in if
            you call this many times).

        Returns
        -------
        int
            Position-within-HP-block in ``{0, 1, 2, ...}``. Clipped to
            ``2`` defensively in case any subject ever has more than 3
            runs of the same HP condition.
        """
        if hp_for_runs is None:
            hp_for_runs = self.get_hpd_locations()

        # Iterate over all (session, run) pairs in chronological order
        # (sessions are scanned in temporal order; runs within a session
        # are also chronological).
        all_pairs = sorted(hp_for_runs.keys())

        # Per-HP running counter.
        per_hp_counter = {}
        positions = {}
        for s, r in all_pairs:
            hp = hp_for_runs[(s, r)]
            pos = per_hp_counter.get(hp, 0)
            positions[(s, r)] = pos
            per_hp_counter[hp] = pos + 1

        if (session, run) not in positions:
            raise KeyError(
                f"(session={session}, run={run}) not found in "
                f"hp_for_runs (subject {self.subject_id}). Available: "
                f"{sorted(positions.keys())}")

        pos = positions[(session, run)]
        # Defensive clip — shouldn't trigger in practice (max 3 per HP).
        return int(min(pos, 2))

    def get_experimental_settings(self, session=1, run=1):
        if self.subject_id < 3:
            yml_file = self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id}' / f'ses-{session+1}' / f'sub-{self.subject_id}_ses-{session+1}_task-ret_sup_run-{run}_expsettings.yml'
        else:
            yml_file = self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id:02d}' / f'ses-{session+1}' / f'run-{run}' / f'sub-{self.subject_id:02d}_ses-{session+1}_task-ret_sup_run-{run}_expsettings.yml'

        # Load yml file
        with open(yml_file, 'r') as file:
            settings = yaml.safe_load(file)

        eccentricity_stimuli = settings['experiment'].get('eccentricity_stimulus')
        size_stimuli = settings['experiment'].get('size_stimuli')
        radius_bar_aperture = eccentricity_stimuli - size_stimuli / 1.8
        radius_bar_aperture
        speed = settings["bar_stimulus"]["speed"]
        bar_width = (radius_bar_aperture * 2) / 8
        fov_size = radius_bar_aperture * 2

        return {
            'eccentricity_stimuli': eccentricity_stimuli,
            'size_stimuli': size_stimuli,
            'radius_bar_aperture': radius_bar_aperture,
            'speed': speed,
            'bar_width': bar_width,
            'fov_size': fov_size
        }


    def get_tr(self, session=1, run=1):
        return 1.6

    def get_n_volumes(self, session=1, run=1):
        return 258
    
    def get_runs(self, session=1):
        # sub-20 ses-1 and sub-24 ses-2 only have 5 runs.
        if (self.subject_id, session) in {(20, 1), (24, 2)}:
            return [1, 2, 3, 4, 5]
        return [1, 2, 3, 4, 5, 6]

    def get_onsets(self, session=1, run=1):
        if self.subject_id < 3:
            onsets = pd.read_csv(self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id}' / f'ses-{session+1}' / f'sub-{self.subject_id}_ses-{session+1}_task-ret_sup_run-{run}_events.tsv', sep='\t')
        else:
            onsets = pd.read_csv(self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id:02d}' / f'ses-{session+1}' / f'run-{run}' /  f'sub-{self.subject_id:02d}_ses-{session+1}_task-ret_sup_run-{run}_events.tsv', sep='\t')

        pulses = onsets[onsets['event_type'] == 'pulse']
        onsets['onset'] = onsets['onset'] - pulses['onset'].min()

        # Add distractor_orientation column = 90 - target_orientation.
        # In the visual search task all distractors share ONE orientation
        # and the target has the OPPOSITE orientation (90 deg apart). The
        # events.tsv only stores ``target_orientation`` (0.0 or 90.0); we
        # derive the distractor orientation as 90 - target_orientation.
        # ``target_orientation`` is filled in on every event row of the
        # trial (trial_start, pre-target, target, response, feedback, iti),
        # so the column-wise transform is well-defined.
        if 'target_orientation' in onsets.columns:
            onsets['distractor_orientation'] = 90.0 - onsets['target_orientation']

        return onsets

    def get_grid_coordinates(self, resolution=100, session=1, run=1,):
        settings = self.get_experimental_settings(session, run)
        radius_bar_aperture = settings['radius_bar_aperture']
        grid_coordinates_1d = np.linspace(-radius_bar_aperture, radius_bar_aperture, resolution)
        grid_coordinates = np.meshgrid(grid_coordinates_1d, grid_coordinates_1d)
        return grid_coordinates

    def get_stimulus(self, session=1, run=1, resolution=100, debug=False):

        settings = self.get_experimental_settings(session, run)

        if debug:
            print("[DEBUG] Sweep parameters:")
            print(f"radius_bar_aperture: {settings['radius_bar_aperture']}")
            print(f"fov_size: {settings['fov_size']}")
            print(f"bar_width: {settings['bar_width']}")
            print(f"speed: {settings['speed']}")

        def draw_bar(grid_coordinates, pos, ori, bar_width):
            # Grid coordinates are 2D arrays
            assert ori in [0, 90], "Orientation must be 0 or 90 degrees"
            bar = np.zeros_like(grid_coordinates[0])
            x, y = grid_coordinates
            if ori == 0:
                bar[np.abs(x - pos) < bar_width / 2] = 1
            elif ori == 90:
                bar[np.abs(y - pos) < bar_width / 2] = 1
            return bar

        tr = self.get_tr(session, run)
        n_volumes = self.get_n_volumes(session, run)
        frametimes = np.arange(tr/2., tr*n_volumes + tr/2., tr)

        onsets = self.get_onsets(session, run)
        bar = onsets[onsets['event_type'].apply(lambda x: x.startswith('bar'))]

        if debug:
            print("[DEBUG] Bar event log:")
            print(bar[['onset', 'event_type']])

        radius_bar_aperture, fov_size, bar_width = settings['radius_bar_aperture'], settings['fov_size'], settings['bar_width']
        speed = settings['speed']

        grid_coordinates = self.get_grid_coordinates(resolution, session, run)
        grid_x, grid_y = grid_coordinates
        mask = np.sqrt(grid_x**2 + grid_y**2) <= radius_bar_aperture
        stimulus = np.zeros((len(frametimes), resolution, resolution))

        ori = 0
        pos = -fov_size - bar_width

        for i, t in enumerate(frametimes):
            if t < bar['onset'].min():
                stimulus[i, :, :] = 0
                continue

            current_state = bar[bar['onset'] < t].iloc[-1]['event_type']
            dt = t - bar[bar['onset'] < t].iloc[-1]['onset']

            if current_state in ['bar_rest', 'bar_break']:
                stimulus[int(t/tr), :, :] = 0
                ori = 0
                pos = 100
            elif current_state in ['bar_right']:
                ori = 0
                pos = -radius_bar_aperture - bar_width / 2 + dt * speed
            elif current_state in ['bar_left']:
                ori = 0
                pos = radius_bar_aperture + bar_width / 2 - dt * speed
            elif current_state in ['bar_up']:
                ori = 90
                pos = -radius_bar_aperture - bar_width / 2 + dt * speed
            elif current_state in ['bar_down']:
                ori = 90
                pos = radius_bar_aperture + bar_width / 2 - dt * speed

            if debug:
                print(f"[DEBUG] Frame {i}: t={t:.2f}, state={current_state}, dt={dt:.2f}, ori={ori}, pos={pos:.2f}")

            stimulus[i, ...] = draw_bar(grid_coordinates, pos, ori, bar_width) * mask

        return stimulus

    def get_extended_grid_coordinates(
        self, resolution=120, session=1, run=1,
        grid_radius=5.0,
    ):
        """Same as get_grid_coordinates but uses a fixed grid_radius
        (default 5°) so the 4°-ring distractor positions fit inside.
        """
        grid_coordinates_1d = np.linspace(-grid_radius, grid_radius, resolution)
        return np.meshgrid(grid_coordinates_1d, grid_coordinates_1d)

    def get_stimulus_with_distractors(
        self, session=1, run=1, resolution=120,
        grid_radius=5.0, distractor_radius=0.4,
        max_distractor_duration=1.5, debug=False,
        distractor_shape='circle',
        distractor_long_side=1.5, distractor_short_side=0.375,
    ):
        """Bar PRF stimulus + distractor pixels at the 4 ring locations.

        Returns a (T, R, R) stimulus array on a grid that extends to
        `grid_radius` (default 5°) — wider than the bar aperture (~3.17°)
        so the distractors at 4° eccentricity fit inside.

        For each trial event with ``distractor_location ∈ {1,3,5,7}`` we
        add a distractor footprint at the corresponding ring location.
        The footprint shape is controlled by ``distractor_shape``:

        - ``'circle'`` (default, BACKWARD-COMPATIBLE): a disk of radius
          ``distractor_radius``.
        - ``'rectangle'``: an oriented rectangle of size
          ``distractor_long_side × distractor_short_side`` whose long
          axis is rotated by the trial's ``distractor_orientation``
          (= ``90 - target_orientation``). 0° → long axis horizontal,
          90° → long axis vertical. This matches the actual
          PsychoPy ``visual.Rect`` rendering of search-array items
          (where ``ori=0`` is horizontal, ``ori=90`` is vertical).

        The footprint's intensity within a given TR is the FRACTION OF
        THE TR during which the distractor was on screen (overlap of
        the on-window with the TR-window divided by the TR duration).
        The on-window starts at the target event onset and extends to
        the next feedback event or, if that's missing/too far, to
        ``t_on + max_distractor_duration``.

        Bar pixel intensities are kept binary (0/1) inside the bar
        aperture; distractor pixels are values in [0, 1].  Where bar and
        distractor overlap (which can't happen since distractors are
        outside the bar aperture) we take the maximum.

        Parameters
        ----------
        session, run : int
            Session and run identifiers.
        resolution : int
            Number of pixels per side for the grid.
        grid_radius : float
            Half-width of the square grid in degrees.
        distractor_radius : float
            Radius of the distractor disk in degrees (used only for
            ``distractor_shape='circle'``).
        max_distractor_duration : float
            Cap on the on-window length (s).  Search-array presentation
            is short (~150 ms) but feedback can come several seconds
            later and the distractors are off long before that.
        distractor_shape : {'circle', 'rectangle'}, default 'circle'
            Distractor footprint. ``'circle'`` matches the original
            implementation bit-for-bit.  ``'rectangle'`` uses the
            ``distractor_long_side / distractor_short_side`` dimensions
            and the trial's ``distractor_orientation``.
        distractor_long_side, distractor_short_side : float
            Long- and short-axis lengths of the rectangle (deg). Defaults
            (1.5, 0.5) approximate the actual search-array rectangles
            (PsychoPy ``visual.Rect(width=size_stimuli,
            height=size_stimuli/4)`` with ``size_stimuli=1.5``, i.e.
            1.5 × 0.375; we round the short side up slightly to 0.5 deg
            to soften aliasing on the 50-pixel grid).
        """
        if distractor_shape not in ('circle', 'rectangle'):
            raise ValueError(
                f"distractor_shape must be 'circle' or 'rectangle', "
                f"got {distractor_shape!r}")
        settings = self.get_experimental_settings(session, run)
        tr = self.get_tr(session, run)
        n_volumes = self.get_n_volumes(session, run)
        # TR window i = [i*tr, (i+1)*tr];  centre = (i + 0.5) * tr.
        frametimes = np.arange(tr / 2., tr * n_volumes + tr / 2., tr)

        # Bar stimulus on the extended grid.
        radius_bar_aperture = settings["radius_bar_aperture"]
        bar_width = settings["bar_width"]
        speed = settings["speed"]
        fov_size = settings["fov_size"]

        grid_x, grid_y = self.get_extended_grid_coordinates(
            resolution=resolution, session=session, run=run,
            grid_radius=grid_radius,
        )
        # Mask for bar (still confined to bar aperture).
        bar_mask = np.sqrt(grid_x ** 2 + grid_y ** 2) <= radius_bar_aperture

        onsets = self.get_onsets(session, run)
        bar = onsets[onsets["event_type"].apply(lambda x: x.startswith("bar"))]

        stimulus = np.zeros((len(frametimes), resolution, resolution),
                            dtype=np.float32)

        # --- Bar stimulus pass (same logic as get_stimulus). ---
        ori = 0
        pos = -fov_size - bar_width

        def draw_bar(pos, ori):
            bar_img = np.zeros_like(grid_x, dtype=np.float32)
            if ori == 0:
                bar_img[np.abs(grid_x - pos) < bar_width / 2] = 1
            elif ori == 90:
                bar_img[np.abs(grid_y - pos) < bar_width / 2] = 1
            return bar_img * bar_mask

        for i, t in enumerate(frametimes):
            if t < bar["onset"].min():
                continue
            current_state = bar[bar["onset"] < t].iloc[-1]["event_type"]
            dt = t - bar[bar["onset"] < t].iloc[-1]["onset"]

            if current_state in ["bar_rest", "bar_break"]:
                continue
            elif current_state == "bar_right":
                ori = 0; pos = -radius_bar_aperture - bar_width / 2 + dt * speed
            elif current_state == "bar_left":
                ori = 0; pos = radius_bar_aperture + bar_width / 2 - dt * speed
            elif current_state == "bar_up":
                ori = 90; pos = -radius_bar_aperture - bar_width / 2 + dt * speed
            elif current_state == "bar_down":
                ori = 90; pos = radius_bar_aperture + bar_width / 2 - dt * speed
            stimulus[i] = np.maximum(stimulus[i], draw_bar(pos, ori))

        # --- Search-array pass: paint ALL 8 items at the actual ring
        # positions during each trial's target-on window. ---
        #
        # Per the experiment code (experiment/stimuli.py:107-188), the
        # search array contains 8 oriented rectangles at positions
        # (eccentricity·cos(i·45°), eccentricity·sin(i·45°)) for i=0..7.
        # Diagonals (1, 3, 5, 7) can be target/distractor; cardinals
        # (0, 2, 4, 6) are filler-only. The TARGET rectangle has
        # ``target_orientation``; all 7 OTHERS (including the color-
        # singleton distractor) share ``90 - target_orientation``.
        # The distractor singleton is a colour outlier — irrelevant
        # for the luminance paradigm tensor.
        targets = onsets[onsets["event_type"] == "target"].sort_values("onset")
        feedback = onsets[onsets["event_type"] == "feedback"].sort_values("onset")
        ecc = settings.get("eccentricity_stimuli", 4.0)
        # 8 ring positions in PsychoPy convention (angle = i·45°,
        # measured CCW from +x axis; +y is up on the screen).
        ring_xy = np.array([
            (ecc * np.cos(np.deg2rad(i * 45.0)),
             ecc * np.sin(np.deg2rad(i * 45.0)))
            for i in range(8)
        ], dtype=np.float64)
        # Distractor disk radius: kept for shape='circle' fallback only.
        # Pre-compute TR window edges for fast overlap math.
        tr_starts = frametimes - tr / 2.
        tr_ends = frametimes + tr / 2.

        def _footprint(cx, cy, ori_deg):
            """One-rectangle (or disk) binary footprint on the grid."""
            if distractor_shape == 'circle':
                return (((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                        < distractor_radius ** 2).astype(np.float32)
            if pd.isna(ori_deg):
                # No orientation info in events.tsv — fall back to disk.
                return (((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                        < distractor_radius ** 2).astype(np.float32)
            # Rotate the pixel grid into the rectangle's local frame.
            # ori_deg is the rectangle's screen orientation in degrees.
            # Inverse rotation: pixel-frame -> rect-frame.
            ang = -float(ori_deg) * np.pi / 180.0
            cos_a = np.cos(ang)
            sin_a = np.sin(ang)
            dx = grid_x - cx
            dy = grid_y - cy
            xr = dx * cos_a + dy * sin_a
            yr = -dx * sin_a + dy * cos_a
            return ((np.abs(xr) <= distractor_long_side / 2.0)
                    & (np.abs(yr) <= distractor_short_side / 2.0)
                    ).astype(np.float32)

        for _, trial in targets.iterrows():
            t_on = trial["onset"]
            after = feedback[feedback["onset"] > t_on]
            t_off = (after.iloc[0]["onset"] if len(after)
                     else t_on + max_distractor_duration)
            t_off = min(t_off, t_on + max_distractor_duration)

            # Fraction of TR overlapping [t_on, t_off].
            overlap = np.clip(
                np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
                0.0, None,
            ) / tr  # in [0, 1]
            active = overlap > 0
            if not active.any():
                continue

            # Per-trial orientations: target gets target_orientation;
            # all 7 others get (90 - target_orientation). Skip if no
            # orientation info → all 8 paint as disks (rare).
            target_ori = trial.get("target_orientation", np.nan)
            other_ori = (90.0 - float(target_ori)
                         if not pd.isna(target_ori) else np.nan)
            target_loc = trial.get("target_location", np.nan)
            try:
                target_loc_int = int(target_loc) if not pd.isna(target_loc) else None
            except (TypeError, ValueError):
                target_loc_int = None

            # Combined footprint for this trial: union over the 8 items.
            combined = np.zeros_like(grid_x, dtype=np.float32)
            for i in range(8):
                cx, cy = ring_xy[i]
                ori_deg = target_ori if i == target_loc_int else other_ori
                fp = _footprint(cx, cy, ori_deg)
                combined = np.maximum(combined, fp)

            contrib = overlap[active, None, None] * combined[None, :, :]
            stimulus[active] = np.maximum(stimulus[active], contrib)

            if debug:
                d_loc = trial.get("distractor_location", np.nan)
                print(
                    f"[DEBUG] trial @ t={t_on:.2f}s, "
                    f"target_loc={target_loc_int}, dist_loc={d_loc}, "
                    f"target_ori={target_ori}, other_ori={other_ori}, "
                    f"on=[{t_on:.2f}, {t_off:.2f}], "
                    f"max overlap fraction={overlap.max():.3f}"
                )

        return stimulus

    # Ring location code → channel index. Channel order MUST stay
    # in sync with `CONDITIONS` in fit_*_af_braincoder.py.
    # 1: upper_right, 3: upper_left, 5: lower_left, 7: lower_right.
    _LOC_TO_CHANNEL = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}

    def _per_bin_indicator(self, session, run, location_col,
                            max_duration, oversampling, trial_filter=None):
        """Compute a per-bin on-fraction indicator at the 4 ring channels.

        Shared machinery behind :meth:`get_dynamic_indicator`,
        :meth:`get_repeat_distractor_indicator` and
        :meth:`get_target_indicator`.

        Args:
            session, run: BIDS indices.
            location_col: events.tsv column to read the ring code from
                (``distractor_location`` or ``target_location``).
            max_duration: cap on each trial's on-duration (s).
            oversampling: temporal sub-binning factor (>=1).
            trial_filter: optional callable ``(prev_loc, this_loc) ->
                bool`` deciding whether to paint a given trial.
                ``prev_loc`` is the previous trial's location code
                (or ``None`` at the start); ``this_loc`` is the
                current one (or ``None`` if missing / 10.0 / not a
                ring location). Default keeps every trial whose
                ``this_loc`` is a valid ring code.

        Returns:
            (n_volumes * oversampling, 4) ``float32`` array of
            per-bin on-fractions in [0, 1].
        """
        if int(oversampling) < 1:
            raise ValueError(f"oversampling must be >= 1, got {oversampling}")
        oversampling = int(oversampling)

        loc_to_channel = self._LOC_TO_CHANNEL
        tr = self.get_tr(session, run)
        n_volumes = self.get_n_volumes(session, run)
        dt = tr / oversampling
        n_bins = n_volumes * oversampling
        frametimes = (np.arange(n_bins, dtype=np.float64) + 0.5) * dt
        tr_starts = frametimes - dt / 2.
        tr_ends = frametimes + dt / 2.

        onsets = self.get_onsets(session, run)
        targets = onsets[onsets["event_type"] == "target"].sort_values("onset")
        feedback = onsets[onsets["event_type"] == "feedback"].sort_values("onset")

        out = np.zeros((n_bins, len(loc_to_channel)), dtype=np.float32)
        prev_loc = None
        for _, trial in targets.iterrows():
            code = trial[location_col]
            this_loc = (code if (not pd.isna(code)
                                  and code in loc_to_channel) else None)

            paint = (trial_filter(prev_loc, this_loc) if trial_filter
                     else (this_loc is not None))
            prev_loc = this_loc  # update history regardless of paint
            if not paint:
                continue

            ch = loc_to_channel[this_loc]
            t_on = trial["onset"]
            after = feedback[feedback["onset"] > t_on]
            t_off = (after.iloc[0]["onset"] if len(after)
                     else t_on + max_duration)
            t_off = min(t_off, t_on + max_duration)
            overlap = np.clip(
                np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
                0.0, None) / dt
            out[:, ch] = np.maximum(out[:, ch], overlap.astype(np.float32))
        return out

    def get_dynamic_indicator(self, session=1, run=1,
                              max_distractor_duration=1.5,
                              oversampling=1):
        """Per-TR distractor-on indicator at each of the 4 ring locations.

        Channel order is
        ``['upper_right', 'upper_left', 'lower_left', 'lower_right']``,
        matching the ``CONDITIONS`` list in fit_*_af_braincoder.py and
        the ``ring_positions`` ordering of
        ``DynamicAttentionFieldPRF2DWithHRF``.

        ``oversampling`` > 1 returns the indicator on a fine grid with
        step ``tr / oversampling`` (shape ``(n_volumes * oversampling,
        4)``).
        """
        return self._per_bin_indicator(
            session, run,
            location_col="distractor_location",
            max_duration=max_distractor_duration,
            oversampling=oversampling)

    def get_repeat_distractor_indicator(self, session=1, run=1,
                                        max_distractor_duration=1.5,
                                        oversampling=1):
        """Per-TR distractor-on indicator restricted to REPEAT trials.

        A trial counts as a "repeat" iff its distractor is at the SAME
        ring location as the immediately preceding trial's distractor
        (both must have a distractor in {1,3,5,7}). The first trial of
        a run is never a repeat. Subtracting this from the full dynamic
        indicator gives the SWITCH-trial indicator.
        """
        def is_repeat(prev_loc, this_loc):
            return this_loc is not None and this_loc == prev_loc
        return self._per_bin_indicator(
            session, run,
            location_col="distractor_location",
            max_duration=max_distractor_duration,
            oversampling=oversampling,
            trial_filter=is_repeat)

    def get_target_indicator(self, session=1, run=1,
                             max_target_duration=1.5,
                             oversampling=1):
        """Per-TR target-on indicator at each of the 4 ring locations.

        Same machinery and channel ordering as
        :meth:`get_dynamic_indicator`, but reads ``target_location``
        instead of ``distractor_location`` — tracks the per-TR overlap
        fraction of the SEARCH TARGET at each of the 4 ring positions.
        Used as a positive-control "phasic capture" channel in the v3 +
        target model.
        """
        return self._per_bin_indicator(
            session, run,
            location_col="target_location",
            max_duration=max_target_duration,
            oversampling=oversampling)

    def get_confounds(self, session=1, run=1, filter_confounds=True,
                       n_acompcorr=10):
        """fmriprep confounds for one (session, run).

        With ``filter_confounds=True`` (default) returns a curated
        subset (~30 cols) used by the cleaning pipeline: dvars, FD,
        top-N a_comp_cor, 6 motion + 6 derivative1 motion, non-steady-
        state, motion_outlier_XX, cosine_XX. With
        ``filter_confounds=False`` returns the full ~159-col tsv.

        NaNs in the kept columns are filled with 0 so they don't
        propagate through ``nilearn.image.clean_img``.
        """
        tsv = (self.bids_folder / 'derivatives' / 'fmriprep'
               / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func'
               / f'sub-{self.subject_id:02d}_ses-{session}_task-search_'
                 f'rec-NORDIC_run-{run}_desc-confounds_timeseries.tsv')
        confounds = pd.read_csv(tsv, sep='\t')
        if not filter_confounds:
            return confounds

        keep = ['dvars', 'framewise_displacement']
        keep += [f'a_comp_cor_{i:02d}' for i in range(n_acompcorr)]
        motion = ['trans_x', 'trans_y', 'trans_z',
                  'rot_x', 'rot_y', 'rot_z']
        keep += motion + [f'{c}_derivative1' for c in motion]
        keep += [c for c in confounds.columns if 'non_steady_state' in c]
        keep += [c for c in confounds.columns if 'motion_outlier' in c]
        keep += [c for c in confounds.columns if 'cosine' in c]
        keep = [c for c in keep if c in confounds.columns]
        return confounds[keep].fillna(0)

    def get_concatenated_bold(self, type='cleaned', voxel_idx=None,
                              crop_to=258):
        """Concatenate BOLD across all this subject's (session, run) pairs.

        One-shot helper for callers who want the whole 12-run timeseries
        as a single ``(T_total, V)`` array — e.g. decoding pipelines or
        cross-run statistics. Each run is loaded through a brain-mask
        ``NiftiMasker`` (so V matches the masker's flat order) and
        cropped to ``crop_to`` TRs (cleaned BOLD is sometimes 259, the
        canonical length is 258).

        Args:
            type: passed through to :meth:`get_bold`
                  (``cleaned`` / ``fmriprep`` / ``prf_regressed_out`` /
                  ``raw``).
            voxel_idx: optional 1D array of masker-flat indices to
                       restrict to. Useful for ROI workflows where the
                       caller already has a per-voxel parameter frame
                       keyed by ``voxel_idx`` (e.g. the V1 warm-start
                       TSV) and just wants the matching timeseries.
            crop_to: max TRs per run.

        Returns:
            ``(T_total, V) float32`` ndarray.
        """
        first_run = self.get_runs(1)[0]
        bold_mask = self.get_bold_mask(session=1, run=first_run)
        masker = input_data.NiftiMasker(mask_img=bold_mask)
        masker.fit()
        chunks = []
        for ses in (1, 2):
            for run in self.get_runs(ses):
                d = masker.transform(self.get_bold(session=ses, run=run,
                                                     type=type))[:crop_to]
                if voxel_idx is not None:
                    d = d[:, voxel_idx]
                chunks.append(d.astype(np.float32))
        return np.vstack(chunks)

    def get_warmstart_pars(self, model, roi='V1', notes_dir=None):
        """Load this subject's warm-start TSV (V1 sandbox output).

        Returns the per-voxel parameter frame written by
        ``notes/figures/talk/fit_prf_warmstart.py`` for the requested
        model. Includes ``voxel_idx`` and ``hemi`` columns so the
        result can be voxel-aligned with
        :meth:`get_concatenated_bold` without further bookkeeping.

        Args:
            model: 1..6 — the PRF model number (m1=Gauss, m2=DoG,
                   m3=Gauss+HRF, m4=DoG+HRF, m5=DN, m6=DN+HRF).
            roi: ROI tag in the TSV filename (today only ``V1``).
            notes_dir: optional override for the TSV directory.
                       Default: ``<repo>/notes/data/``.

        Returns:
            ``DataFrame`` with at least ``x, y, sd, ..., r2, hemi,
            voxel_idx`` (parameter set depends on ``model``).
        """
        if notes_dir is None:
            # retsupp/utils/data.py -> retsupp/utils -> retsupp -> repo
            notes_dir = Path(__file__).resolve().parents[2] / 'notes' / 'data'
        tsv = (Path(notes_dir) / f'prf_warmstart_m{model}_{roi}_'
               f'sub-{self.subject_id:02d}.tsv')
        if not tsv.exists():
            raise FileNotFoundError(tsv)
        return pd.read_csv(tsv, sep='\t')

    def get_bold(self, session=1, run=1, type='cleaned', return_image=True):

        if type == 'fmriprep':
            fn = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_space-T1w_desc-preproc_bold.nii.gz'
        elif type == 'raw':
            fn = self.bids_folder / 'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_run-{run}_bold.nii.gz'
        elif type == 'cleaned':
            fn = self.bids_folder / 'derivatives' / 'cleaned' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_desc-cleaned_run-{run}_bold.nii.gz'
        elif type == 'prf_regressed_out':
            # sub-02_ses-1_run-1_task-prf_cleaned_regressed.nii.gz
            fn = self.bids_folder / 'derivatives' / 'prf_regressed_out' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_run-{run}_task-prf_cleaned_regressed.nii.gz'
        else:
            raise ValueError("Type must be 'fmriprep', 'raw', 'cleaned', or 'nordic', 'prf_regressed_out")

        if return_image:
            return image.load_img(fn)
        else:
            return fn

    def get_gm_mask(self, bold_space: bool = True, threshold: float = 0.5):
        """Return a gray-matter mask from fMRIprep's GM probseg.

        Resamples ``derivatives/fmriprep/sub-XX/ses-1/anat/sub-XX_ses-1_label-GM_probseg.nii.gz``
        into BOLD space (when ``bold_space=True``) and thresholds at
        ``probseg >= threshold``. Returns a binary NIfTI image.
        """
        gm_path = (self.bids_folder / 'derivatives' / 'fmriprep'
                   / f'sub-{self.subject_id:02d}' / 'ses-1' / 'anat'
                   / f'sub-{self.subject_id:02d}_ses-1_label-GM_probseg.nii.gz')
        gm = image.load_img(gm_path)
        if bold_space:
            gm = image.resample_to_img(gm, self.get_bold_mask(),
                                        interpolation='linear',
                                        force_resample=True, copy_header=True)
        data = (image.get_data(gm) >= threshold).astype('uint8')
        return image.new_img_like(gm, data)

    def get_bold_mask(self, session=1, run=1, return_masker=False):
        fn = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_space-T1w_desc-brain_mask.nii.gz'

        if return_masker:
            return input_data.NiftiMasker(mask_img=fn)
        # Retry-on-not-found: shared NFS sometimes hands compute nodes
        # a stale view of the BIDS dir; the file exists but isn't
        # visible until the mount cache refreshes.
        import time
        for attempt in range(4):
            try:
                return image.load_img(fn)
            except (FileNotFoundError, ValueError) as e:
                if 'File not found' not in str(e) and not isinstance(
                        e, FileNotFoundError):
                    raise
                wait = 15 * (attempt + 1)
                print(f'  [retry {attempt+1}/4] mask not yet visible '
                      f'on this node; sleeping {wait}s', flush=True)
                time.sleep(wait)
        return image.load_img(fn)  # final attempt, let it raise

    def get_surf_info(self):
        info = {'L':{}, 'R':{}}

        for hemi in ['L', 'R']:

            fs_hemi = {'L':'lh', 'R':'rh'}[hemi]



            if self.subject_id < 3:
                info[hemi]['inner'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'anat' / f'sub-{self.subject_id:02d}_hemi-{hemi}_white.surf.gii'
                info[hemi]['outer'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'anat' / f'sub-{self.subject_id:02d}_hemi-{hemi}_pial.surf.gii'
                info[hemi]['thickness'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'anat' / f'sub-{self.subject_id:02d}_hemi-{hemi}_thickness.shape.gii'
            else:
                info[hemi]['inner'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'ses-1' / 'anat' / f'sub-{self.subject_id:02d}_ses-1_hemi-{hemi}_white.surf.gii'
                info[hemi]['outer'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'ses-1' / 'anat' / f'sub-{self.subject_id:02d}_ses-1_hemi-{hemi}_pial.surf.gii'
                info[hemi]['thickness'] = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'ses-1' / 'anat' / f'sub-{self.subject_id:02d}_ses-1_hemi-{hemi}_thickness.shape.gii'


            for key in info[hemi]:
                assert(info[hemi][key]).exists(), f'{info[hemi][key]} does not exist'

        return info

    def get_prf_parameter_labels(self, model=1):
        labels = ['x', 'y', 'sd', 'amplitude', 'baseline', 'r2', 'theta', 'ecc']

        if model == 2:
            labels += ['srf_size', 'srf_amplitude']
        elif model == 3:
            labels += ['hrf_delay', 'hrf_dispersion']
        elif model in [4,7,8,9,10,11]:
            labels += ['srf_size', 'srf_amplitude', 'hrf_delay', 'hrf_dispersion']
        elif model == 6:
            # Get rid of amplitude parmameter
            labels.pop(labels.index('amplitude'))
            labels.pop(labels.index('baseline'))

            labels += ['rf_amplitude', 'srf_amplitude', 'srf_size',
                       'neural_baseline', 'surround_baseline',
                       'bold_baseline']

            labels += ['hrf_delay', 'hrf_dispersion']
        elif model == 5:
            # Same parameter set as model 6 but WITHOUT flex HRF.
            labels.pop(labels.index('amplitude'))
            labels.pop(labels.index('baseline'))

            labels += ['rf_amplitude', 'srf_amplitude', 'srf_size',
                       'neural_baseline', 'surround_baseline',
                       'bold_baseline']
        elif model == 1:
            pass
        else:
            raise ValueError(f'Unknown model parameters for model: {model}')

        return labels

    def get_prf_parameters_volume(self, model=1, type='mean', return_images=True):
        """
        Extract PRF parameter images for this subject.
        Args:
            model (int): Model number
            runwise (bool): If True, return images for each session/run. If False, return one image per parameter.
        Returns:
            If runwise is False: pd.Series with parameter labels as index and NiftiImages as values.
            If runwise is True: pd.DataFrame with columns as parameter labels, index as session/run pairs, and values as NiftiImages.
        """

        param_labels = self.get_prf_parameter_labels(model=model)

        if type == 'mean':
            base_dir = self.bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{self.subject_id:02d}'
            images = {}
            for par in param_labels:
                img_path = base_dir / f'sub-{self.subject_id:02d}_desc-{par}.nii.gz'
                images[par] = nib.load(str(img_path))
            
            if return_images:
                return pd.Series(images)
            data = {par: self._extract_param_arr(images[par], roi=None)
                    for par in param_labels}
            return pd.DataFrame(data)

        elif type=='runwise':

            base_dir = self.bids_folder / 'derivatives' / 'prf_runfit' / f'model{model}' / f'sub-{self.subject_id:02d}'
            data = []
            index = []
            for ses in [1, 2]:
                ses_dir = base_dir / f'ses-{ses}'
                for run in [1, 2, 3, 4, 5, 6]:
                    row = {}
                    for par in param_labels:
                        img_path = ses_dir / f'sub-{self.subject_id:02d}_ses-{ses}_run-{run}_desc-{par}.nii.gz'
                        row[par] = nib.load(str(img_path))
                    data.append(row)
                    index.append((ses, run))
            df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['session', 'run']))
            df.columns.name = 'parameter'
            return df
        elif type == 'conditionwise':
            # Four conditions: upper_left, upper_right, lower_left, lower_right.
            # Layout is nested:
            #   prf_conditionfit/model{N}/sub-XX/condition-<cond>/sub-XX_desc-<par>.nii.gz
            conditions = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
            base_dir = self.bids_folder / 'derivatives' / 'prf_conditionfit' / f'model{model}' / f'sub-{self.subject_id:02d}'
            data = []
            index = []
            for cond in conditions:
                cond_dir = base_dir / f'condition-{cond}'
                row = {}
                for par in param_labels:
                    img_path = cond_dir / f'sub-{self.subject_id:02d}_desc-{par}.nii.gz'
                    row[par] = nib.load(str(img_path))
                data.append(row)
                index.append(cond)
            df = pd.DataFrame(data, index=pd.Index(index, name='condition'))
            df.columns.name = 'parameter'
            return df
        else:
            raise ValueError("type must be 'mean', 'runwise', or 'conditionwise'")

    def get_prf_roi_pars(self, roi: str, model: int = 4,
                         params=None, cache: bool = True,
                         force_refresh: bool = False) -> pd.DataFrame:
        """Per-voxel PRF parameters restricted to one ROI, cached as NPZ.

        For each `(subject, model, roi)`, extracts the chosen parameters
        from the volumetric NIfTIs and masks them to the ROI. Returns a
        DataFrame with one row per voxel and columns = parameter names.

        Caching: if `cache=True`, writes/reads
        `<bids>/derivatives/prf_roi_cache/model{M}/sub-XX/{roi}.npz`.
        Cache is reused when the latest source-NIfTI mtime is ≤ the
        mtime recorded in the cache; otherwise re-extracted.

        Args:
            roi: ROI name accepted by `get_retinotopic_roi`
                 (e.g. 'V1', 'V1_L', 'V3AB', 'IPS').
            model: PRF model index (default 4).
            params: which parameter NIfTIs to extract. Default is
                the union of {'x', 'y', 'sd', 'r2'} and the parameter
                labels reported by `get_prf_parameter_labels(model)`.
            cache: whether to read/write the on-disk cache.
            force_refresh: if True, ignore the cache and re-extract.

        Returns:
            pd.DataFrame indexed 0..N-1 (one row per ROI voxel), with
            one column per requested parameter (float32).
        """
        from time import time
        base_dir = (self.bids_folder / 'derivatives' / 'prf'
                    / f'model{model}' / f'sub-{self.subject_id:02d}')
        if params is None:
            base_pars = ('x', 'y', 'sd', 'r2')
            label_pars = tuple(self.get_prf_parameter_labels(model=model))
            params = list(dict.fromkeys(base_pars + label_pars))

        src_paths = [base_dir
                     / f'sub-{self.subject_id:02d}_desc-{par}.nii.gz'
                     for par in params]
        missing = [p for p in src_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"PRF NIfTIs missing for sub-{self.subject_id:02d} "
                f"model {model}: {[p.name for p in missing[:3]]}")
        src_mtime = max(p.stat().st_mtime for p in src_paths)
        # Neuropythy ROI atlas mtime — invalidates the cache when ROIs
        # are regenerated (e.g. after a re-run of register_retinotopy).
        roi_mtime = self.neuropythy_mtime(model=model)
        fresh_threshold = max(src_mtime, roi_mtime)

        # Cache fast-path: if the npz is fresh and covers the requested
        # params, return without ever touching the ROI atlas
        # (get_retinotopic_roi loads + resamples varea — ~1s).
        cache_path = (self.bids_folder / 'derivatives' / 'prf_roi_cache'
                      / f'model{model}' / f'sub-{self.subject_id:02d}'
                      / f'{roi}.npz')
        if cache and not force_refresh and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as d:
                cached_mtime = float(d['source_mtime'])
                cached_roi_mtime = float(d['roi_mtime']) if 'roi_mtime' in d.files else 0.0
                cached_params = [str(p) for p in d['params']]
                # Treat cache as valid only if it covers BOTH the source
                # NIfTIs and the current neuropythy ROI atlas.
                if (cached_mtime >= src_mtime
                        and cached_roi_mtime >= roi_mtime
                        and set(params).issubset(cached_params)):
                    return pd.DataFrame(
                        {p: d[f'col_{p}'] for p in params})

        # Cache miss → load mask + extract.
        roi_img = self.get_retinotopic_roi(roi=roi, bold_space=True)
        roi_mask = np.asarray(roi_img.get_fdata(), dtype=bool)
        if not roi_mask.any():
            return pd.DataFrame({p: np.array([], dtype=np.float32)
                                 for p in params})
        flat = roi_mask.ravel()
        cols = {}
        for par, path in zip(params, src_paths):
            arr = nib.load(str(path)).get_fdata().ravel()[flat]
            cols[par] = arr.astype(np.float32)
        df = pd.DataFrame(cols)

        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path,
                     source_mtime=np.float64(src_mtime),
                     roi_mtime=np.float64(roi_mtime),
                     params=np.array(params),
                     **{f'col_{p}': cols[p] for p in params})
        return df

    def _extract_param_arr(self, fn, roi):
        if roi is None:
            masker = self.get_bold_mask(return_masker=True)
        else:
            roi_mask = self.get_retinotopic_roi(roi=roi, bold_space=True)
            masker = input_data.NiftiMasker(mask_img=roi_mask)
        return masker.fit_transform(fn).squeeze()

    def get_prf_parameters_surface(self, model=1, space='fsnative'):

        parameters = self.get_prf_parameter_labels(model=model)

        output = []

        for par in parameters:
            tmp = []
            for hemi in ['L', 'R']:
                fn = self.bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{self.subject_id:02d}' / f'sub-{self.subject_id:02d}_desc-{par}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii'
                surf_data = surface.load_surf_data(fn).squeeze()
                tmp.append(pd.Series(surf_data, name=(par), index=pd.Index(range(surf_data.shape[0]), name='vertex')))
                # output.append(pd.Series(surface.load_surf_data(fn), name=(par, hemi)))
            output.append(pd.concat(tmp, axis=0, keys=['L', 'R'], names=['hemi']))

        output = pd.concat(output, axis=1)

        return output
    
    def get_hemisphere_mask(self, hemi, bold_space=False):
        """
        Returns a boolean mask for the specified hemisphere ('L' or 'R') 
        based on aparc+aseg.mgz codes: 1021 (left), 2029 (right).
        """
        aseg_file = (
            self.bids_folder
            / "derivatives"
            / "fmriprep"
            / "sourcedata"
            / "freesurfer"
            / f"sub-{self.subject_id:02d}"
            / "mri"
            / "aparc+aseg.mgz"
        )
        aseg_img = image.load_img(str(aseg_file))
    
        if hemi.upper() == "L":
            mask = image.math_img('(aseg >= 1000) & (aseg < 2000)', aseg=aseg_img)
        elif hemi.upper() == "R":
            mask = image.math_img('(aseg >= 2000) & (aseg < 3000)', aseg=aseg_img)
        else:
            raise ValueError("hemi must be 'L' or 'R'")

        mask = nib.Nifti1Image(mask.get_fdata(), affine=mask.affine)

        if bold_space:
            mask = image.resample_to_img(mask, target_img=self.get_bold_mask(), interpolation='nearest', force_resample=True)

        return mask

    def get_t1w(self):
        t1w = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / 'anat' / f'sub-{self.subject_id:02d}_desc-preproc_T1w.nii.gz'

        if t1w.exists():
            return image.load_img(str(t1w))
        else:
            for session in [1,2]:
                t1w = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'anat' / f'sub-{self.subject_id:02d}_ses-{session}_desc-preproc_T1w.nii.gz'
                if t1w.exists():
                    return image.load_img(str(t1w))
        raise FileNotFoundError(f"T1w image not found: {t1w}")   

    def get_retinotopic_labels(self):
        return {
            1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO1', 6: 'VO2', 7: 'LO1', 8: 'LO2',
            9: 'TO1', 10: 'TO2', 11: 'V3A', 12: 'V3B',}
        #13: 'IPS0', 14: 'IPS1', 15: 'IPS2',
         #   16: 'IPS3', 17: 'IPS4', 18: 'IPS5', 19: 'SPL1', 20: 'FEF'
        # 


    @staticmethod
    def safe_inverse_transform_save(masker, values, path, dtype=np.float32):
        """Inverse-transform `values` into a NIfTI and save it safely.

        Wraps the uint8-quantization trap: nilearn maskers inherit the
        mask's dtype (typically uint8) and let nibabel auto-pick an
        scl_slope, which quantizes parameter values to 256 distinct
        levels across the brain. Forcing dtype + clearing the scale
        factor before write avoids that. See CLAUDE.md §"NIfTI dtype
        trap".
        """
        img = masker.inverse_transform(values)
        img.set_data_dtype(dtype)
        img.header.set_slope_inter(slope=1, inter=0)
        img.to_filename(str(path))
        return img

    def neuropythy_mtime(self, model: int = 4) -> float:
        """Latest mtime among the neuropythy `inferred_*.mgz` files.

        Returns the max mtime across ``inferred_varea/eccen/angle/sigma.mgz``
        in the per-model snapshot dir (with freesurfer fallback). Used by
        ROI-keyed caches (`get_prf_roi_pars`, the per-ROI entries in
        `desc-p_signal.json`) to detect when a neuropythy re-run has
        invalidated them. Returns 0.0 if no MGZs are found (no caching
        invalidation triggered).
        """
        mtimes = []
        for name in ('inferred_varea.mgz', 'inferred_eccen.mgz',
                     'inferred_angle.mgz', 'inferred_sigma.mgz'):
            try:
                p = self._neuropythy_mgz_path(name, 'mri', model=model)
                if p.exists():
                    mtimes.append(p.stat().st_mtime)
            except Exception:
                pass
        return max(mtimes) if mtimes else 0.0

    def _neuropythy_mgz_path(self, name, location, model=4):
        """Resolve an `inferred_*.mgz` path with per-model snapshot fallback.

        Reads from ``derivatives/neuropythy/model{N}/sub-XX/{location}/`` when
        available; otherwise falls back to the canonical freesurfer subject
        dir (most recent neuropythy run, typically model 4).
        """
        per_model = (self.bids_folder / 'derivatives' / 'neuropythy'
                     / f'model{model}' / f'sub-{self.subject_id:02d}'
                     / location / name)
        if per_model.exists():
            return per_model
        return (self.bids_folder / 'derivatives' / 'fmriprep' / 'sourcedata'
                / 'freesurfer' / f'sub-{self.subject_id:02d}' / location / name)

    def get_retinotopic_atlas(self, bold_space=False, model=4):
        varea_file = self._neuropythy_mgz_path('inferred_varea.mgz', 'mri',
                                                model=model)
        varea_img = image.load_img(str(varea_file))

        # Make Nifti1Image out of MGZImage
        varea_img = nib.Nifti1Image(varea_img.get_fdata(), affine=varea_img.affine)

        if bold_space:
            func_mask = self.get_bold_mask()
            varea_img = image.resample_to_img(varea_img, target_img=func_mask, interpolation='nearest', force_resample=True, copy_header=True)

        return varea_img

    def get_wang_labels(self):
        """Return the Wang 2015 maximum-probability retinotopy atlas labels.

        Label values 1..25 correspond to the named visual areas in the
        Wang 2015 max-probability atlas. V1d/V1v etc. are split here
        (dorsal/ventral); :meth:`get_retinotopic_roi` aliases such as
        ``'IPS'`` collapse the IPS sub-fields into a union mask.
        """
        return {
            1: 'V1v', 2: 'V1d', 3: 'V2v', 4: 'V2d', 5: 'V3v', 6: 'V3d',
            7: 'hV4', 8: 'VO1', 9: 'VO2', 10: 'PHC1', 11: 'PHC2',
            12: 'TO2', 13: 'TO1', 14: 'LO2', 15: 'LO1', 16: 'V3B', 17: 'V3A',
            18: 'IPS0', 19: 'IPS1', 20: 'IPS2', 21: 'IPS3', 22: 'IPS4',
            23: 'IPS5', 24: 'SPL1', 25: 'FEF',
        }

    def get_wang_atlas(self, bold_space=False):
        """Return the Wang 2015 atlas (label values 1..25) as a NIfTI image.

        Loads ``mri/wang15_atlas.mgz`` from the subject's freesurfer
        directory. If ``bold_space=True``, resamples nearest-neighbour
        into the BOLD (functional) grid.
        """
        wang_file = (
            self.bids_folder
            / "derivatives"
            / "fmriprep"
            / "sourcedata"
            / "freesurfer"
            / f"sub-{self.subject_id:02d}"
            / "mri"
            / "wang15_atlas.mgz"
        )
        wang_img = image.load_img(str(wang_file))

        # Make Nifti1Image out of MGZImage
        wang_img = nib.Nifti1Image(wang_img.get_fdata(), affine=wang_img.affine)

        if bold_space:
            func_mask = self.get_bold_mask()
            wang_img = image.resample_to_img(wang_img, target_img=func_mask, interpolation='nearest', force_resample=True, copy_header=True)

        return wang_img

    # ROIs sourced from the Wang 2015 atlas (rather than the Benson
    # neuropythy ``inferred_varea.mgz`` atlas). Used by
    # :meth:`get_retinotopic_roi` to dispatch.
    _WANG_ONLY_ROIS = {
        'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5',
        'SPL1', 'FEF',
    }
    # Aliases that map onto a union of Wang labels.
    _WANG_ALIASES = {
        'IPS': ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5'],
    }
    # Aliases that map onto a union of Benson labels.
    _BENSON_ALIASES = {
        'V3AB': ['V3A', 'V3B'],
        'LO': ['LO1', 'LO2'],
        'TO': ['TO1', 'TO2'],
        'VO': ['VO1', 'VO2'],
    }

    def get_r2_fdr_threshold(self, model: int, alpha: float = 0.05,
                              roi: str = 'BRAIN',
                              prf_base_dir: str = 'prf',
                              force: bool = False) -> float:
        """Cached tail-FDR R² threshold from the logit-Gaussian mixture.

        ``roi='BRAIN'`` → whole BOLD mask; ``roi='GM'`` → fmriprep GM
        probseg ≥ 0.5; any other ROI name → per-retinotopic-ROI mixture
        (fit + cached on first call). Thin wrapper around
        :func:`retsupp.modeling.compute_r2_mixture.r2_fdr_threshold` —
        see there for caching paths and PDF side-effects.

        Use this in place of the older BH-FDR-on-F-test logic: the
        mixture matches the empirical R² distribution rather than
        assuming an F-distribution null.
        """
        from retsupp.modeling.compute_r2_mixture import r2_fdr_threshold
        return r2_fdr_threshold(
            self.subject_id, model, self.bids_folder, alpha=alpha,
            roi=roi, prf_base_dir=prf_base_dir, force=force)

    def get_r2_threshold(self, model: int, roi: str,
                          posterior: float = 0.5,
                          prf_base_dir: str = 'prf') -> float:
        """R² threshold for (subject, ROI) at which the 2-component
        logit-Gaussian posterior P(signal | R²) first exceeds
        ``posterior`` (moving right from the noise mode).

        Reads the per-(subject, ROI) mixture parameters from the JSON
        sidecar produced by ``compute_r2_mixture.py``. Returns ``np.nan``
        if no sidecar exists for this (subject, ROI) cell.

        ``posterior=0.5`` ≈ majority-signal (lenient); 0.95 conservative.
        """
        import json
        from scipy.stats import norm
        sidecar = (self.bids_folder / 'derivatives' / prf_base_dir
                   / f'model{model}' / f'sub-{self.subject_id:02d}'
                   / f'sub-{self.subject_id:02d}_desc-p_signal.json')
        if not sidecar.exists():
            return np.nan
        with open(sidecar) as fh:
            summary = json.load(fh)
        info = summary.get(roi)
        if not info or 'signal_mu' not in info:
            return np.nan
        # Posterior P(signal | R²) under the logit-Gaussian mixture: walk
        # right from the noise mode on the logit-scale grid; pick the
        # first crossing of P(signal) ≥ posterior.
        z_grid = np.linspace(info['noise_mu'] - 4 * info['noise_sigma'],
                              info['signal_mu'] + 6 * info['signal_sigma'],
                              4000)
        p_n = (info['noise_weight']
               * norm.pdf(z_grid, info['noise_mu'], info['noise_sigma']))
        p_s = (info['signal_weight']
               * norm.pdf(z_grid, info['signal_mu'], info['signal_sigma']))
        p_sig = p_s / (p_n + p_s + 1e-30)
        start = int(np.searchsorted(z_grid, info['noise_mu']))
        cr = np.where(p_sig[start:] >= posterior)[0]
        if len(cr) == 0:
            return float('inf')
        return float(1.0 / (1.0 + np.exp(-z_grid[start + cr[0]])))

    def get_retinotopic_roi(self, roi=None, bold_space=False, model=4):
        """
        Returns a mask image for the specified retinotopic ROI (e.g., 'V1', 'V2', etc.).
        If hemi is 'L' or 'R', restricts to that hemisphere.

        Sources the atlas based on the ROI name:
        - Benson (``inferred_varea.mgz``) for V1, V2, V3, hV4, VO1, VO2,
          LO1, LO2, TO1, TO2, V3A, V3B, plus the alias ``V3AB`` (=V3A∪V3B).
          ``model`` selects which PRF-fit basis the Benson atlas was
          derived from (default 4; pass 6 to use the DN+HRF run).
        - Wang 2015 (``wang15_atlas.mgz``) for IPS0..IPS5, SPL1, FEF,
          plus the alias ``IPS`` (=IPS0∪..∪IPS5). Wang is model-independent.

        ROI names ending in ``_L`` or ``_R`` are restricted to the
        corresponding hemisphere via :meth:`get_hemisphere_mask`.
        """
        if roi is None:
            atlas = self.get_retinotopic_atlas(bold_space=bold_space, model=model)
            labels = self.get_retinotopic_labels()
            return atlas, [label for _, label in sorted(labels.items())]

        if roi.endswith('_L'):
            hemi = 'L'
            roi = roi[:-2]
        elif roi.endswith('_R'):
            hemi = 'R'
            roi = roi[:-2]
        else:
            hemi = None

        # Decide which atlas to source from.
        use_wang = (roi in self._WANG_ONLY_ROIS) or (roi in self._WANG_ALIASES)
        if use_wang:
            atlas = self.get_wang_atlas(bold_space=bold_space)
            labels = self.get_wang_labels()
            aliases = self._WANG_ALIASES
            atlas_name = "Wang"
        else:
            atlas = self.get_retinotopic_atlas(bold_space=bold_space, model=model)
            labels = self.get_retinotopic_labels()
            aliases = self._BENSON_ALIASES
            atlas_name = "Benson"

        component_rois = aliases.get(roi, [roi])
        name_to_idx = {v.lower(): k for k, v in labels.items()}
        indices = []
        for r in component_rois:
            idx = name_to_idx.get(r.lower())
            if idx is None:
                raise ValueError(
                    f"ROI '{r}' not found in {atlas_name} label names.")
            indices.append(idx)
        cond = ' | '.join(f'(img == {i})' for i in indices)
        roi_mask = image.math_img(cond, img=atlas)

        # Wang ROIs (IPS0..FEF) can border Benson territory — empirically
        # only Wang IPS0 overlaps Benson V3B by ~5-15% across subjects,
        # but the guard is cheap and makes the per-ROI fits provably
        # non-overlapping with Benson-sourced fits.
        if use_wang:
            benson_atlas = self.get_retinotopic_atlas(
                bold_space=bold_space, model=model)
            roi_mask = image.math_img("roi * (benson == 0)",
                                       roi=roi_mask, benson=benson_atlas)

        if hemi is not None:
            hemi_mask = self.get_hemisphere_mask(hemi, bold_space=bold_space)
            # Convert boolean array to image with same affine/shape as roi_mask
            roi_mask = image.math_img("roi * hemi", roi=roi_mask, hemi=hemi_mask)

        return roi_mask

    def get_prf_predictions(self, model=1, type='mean', session=None, run=None, return_image=True):

        if type not in ['mean', 'run']:
            raise NotImplementedError("Only 'mean' type is implemented for PRF predictions.")

        
        if type == 'mean':
            fn = self.bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{self.subject_id:02d}' / f'sub-{self.subject_id:02d}_desc-pred.nii.gz'
        
        if type == 'run':
            if (session is None) or (run is None):
                raise ValueError("For 'run' type, both session and run must be specified.")
            fn = self.bids_folder / 'derivatives' / 'prf_runfit' / f'model{model}' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / f'sub-{self.subject_id:02d}_ses-{session}_run-{run}_desc-pred.nii.gz'

        if return_image:
            return image.load_img(str(fn))
        else:
            masker = self.get_bold_mask(return_masker=True)
            data = masker.fit_transform(image.load_img(str(fn)))
            return pd.DataFrame(data, index=pd.Index(range(data.shape[0]), name='time'), columns=pd.Index(range(data.shape[1], name='voxel')))


    def get_inferred_pars_volume(self, return_images=True, model=4):
        import pandas as pd

        par_labels = ['angle', 'eccen', 'sigma', 'varea']
        results = {}
        for par in par_labels:
            results[par] = nib.load(self._neuropythy_mgz_path(
                f'inferred_{par}.mgz', 'mri', model=model))

        if return_images:
            return pd.Series(results)
        else:
            data = {par: self._extract_param_arr(results[par], roi=None) for par in par_labels}
            return pd.DataFrame(data)

    def get_inferred_prf_pars_surf(self, model=4):
        from nibabel.freesurfer.io import read_morph_data
        import numpy as np
        import pandas as pd
        from itertools import product

        par_labels = ['angle', 'eccen', 'sigma', 'varea']
        hemis = ['L', 'R']
        fs_hemi = {'L': 'lh', 'R': 'rh'}

        # Load data for each hemisphere
        dfs = []
        for hemi in hemis:
            df = pd.DataFrame({
                par: read_morph_data(self._neuropythy_mgz_path(
                    f'{fs_hemi[hemi]}.inferred_{par}', 'surf', model=model)
                ).squeeze().astype(np.float32)
                for par in par_labels
            })
            n_vertices = len(df)
            # Create a MultiIndex for this hemisphere
            df.index = pd.MultiIndex.from_product(
                [[hemi], range(n_vertices)],
                names=['hemi', 'vertex']
            )
            dfs.append(df)

        # Concatenate along rows
        results = pd.concat(dfs, axis=0)

        results['roi'] = results['varea'].map(self.get_retinotopic_labels()).fillna('None')

        # Set all 0's to nan
        results['varea'] = results['varea'].replace(0, np.nan)

        return results

    def get_wang_labels_surf(self, prob_thresh=None):
        """Wang 2015 atlas, sampled on the fsnative surface.

        With ``prob_thresh=None`` (default), uses the discrete max-prob
        labels from ``surf/{lh,rh}.wang15_mplbl.mgz`` — same as Wang's
        published max-prob atlas.

        With ``prob_thresh`` ∈ [0, 1], instead reads the full per-area
        probability maps (``wang15_fplbl.mgz``, shape ``(25, n_vertices)``),
        takes argmax across all 25 areas, and labels the vertex only if
        that winning probability exceeds ``prob_thresh``. Lower threshold
        → more vertices labeled. Useful for the small frontal/parietal
        areas (FEF, SPL1, IPS5) where mplbl is conservative — at
        ``prob_thresh=0.3`` FEF grows ~8×, SPL1 ~2×, while IPS0..IPS3
        barely change.

        Returns a DataFrame indexed by ``(hemi, vertex)`` with integer
        label in ``varea`` (1..25, 0 → NaN) and the named area in ``roi``.
        """
        import numpy as np
        import pandas as pd

        fs_hemi = {'L': 'lh', 'R': 'rh'}
        fs_dir = (self.bids_folder / 'derivatives' / 'fmriprep'
                  / 'sourcedata' / 'freesurfer'
                  / f'sub-{self.subject_id:02d}' / 'surf')

        dfs = []
        for hemi in ['L', 'R']:
            if prob_thresh is None:
                arr = nib.load(str(fs_dir / f'{fs_hemi[hemi]}.wang15_mplbl.mgz')
                               ).get_fdata().squeeze().astype(np.float32)
            else:
                # fplbl: (1, 1, 25, n_vertices) → (25, n_vertices) after squeeze
                fp = nib.load(str(fs_dir / f'{fs_hemi[hemi]}.wang15_fplbl.mgz')
                              ).get_fdata().squeeze()
                winner = fp.argmax(axis=0)
                max_prob = fp[winner, np.arange(fp.shape[1])]
                arr = (winner + 1).astype(np.float32)
                arr[max_prob < prob_thresh] = 0.0

            df = pd.DataFrame({'varea': arr})
            df.index = pd.MultiIndex.from_product(
                [[hemi], range(len(df))], names=['hemi', 'vertex'])
            dfs.append(df)
        out = pd.concat(dfs, axis=0)
        out['roi'] = out['varea'].map(self.get_wang_labels()).fillna('None')
        out['varea'] = out['varea'].replace(0, np.nan)
        return out

    def get_eccentric_roi(self, roi, quadrant, bold_space=True, return_masker=True):
        """
        Returns a mask image for the specified eccentricity ROI and quadrant.
        """

        if not bold_space:
            raise NotImplementedError("Only bold_space=True is implemented for eccentricity ROIs.")

        if roi.endswith('_L'):
            assert quadrant in ['lower_right', 'upper_right'], "Left hemisphere can only have right visual field quadrants."
        elif roi.endswith('_R'):
            assert quadrant in ['lower_left', 'upper_left'], "Right hemisphere can only have left visual field quadrants."
        else:
            raise ValueError("ROI must end with '_L' or '_R' to specify hemisphere.")

        fn = self.bids_folder / 'derivatives' / 'stimulus_rois' / f'sub-{self.subject_id:02d}' / f'sub-{self.subject_id:02d}_desc-{roi}_{quadrant}_roi.nii.gz'

        img = image.load_img(fn)

        if bold_space:
            func_mask = self.get_bold_mask()
            img = image.resample_to_img(img, target_img=func_mask, interpolation='nearest', force_resample=True, copy_header=True)
        
        if return_masker:
            return input_data.NiftiMasker(mask_img=img)
        else:
            return img

    def get_conditionwise_summary_prf_pars(self, model=8, ecc_distractor=4.0):

        from retsupp.utils import rotate_x_y

        par_labels = ['x', 'y', 'sd', 'amplitude', 'ecc', 'r2',
                    'srf_size', 'srf_amplitude', 'hrf_delay', 'hrf_dispersion']

        df = pd.read_csv(
            self.bids_folder / f'derivatives/prf_summaries.conditionwise/model{model}/sub-{self.subject_id:02d}/sub-{self.subject_id:02d}_model-{model}_prf_voxels.tsv',
            sep='\t',
            index_col=[0, 1, 2]  # roi, voxel, condition  (roi includes _L/_R)
        )

        mean_model_label = 4

        mean_model = pd.read_csv(
            self.bids_folder / f'derivatives/prf_summaries/model{mean_model_label}/sub-{self.subject_id:02d}/sub-{self.subject_id:02d}_model-{mean_model_label}_prf_voxels.tsv',
            sep='\t',
            index_col=[0, 1]
        )

        df = df.join(mean_model[par_labels], on=['roi', 'voxel'], rsuffix='_mean_model')

        df = df.reset_index()
        roi_mapping = {'LO1':'LO', 'LO2':'LO', 'V3A':'V3AB', 'V3B':'V3AB', 'TO1':'TO', 'TO2':'TO', 'VO1':'VO', 'VO2':'VO'}
        df['roi_base'] = df['roi'].str.replace(r'_(L|R)$', '', regex=True)
        df['roi_base'] = df['roi_base'].replace(roi_mapping)

        # Collision-free, and you still have original roi in the index
        df = df.set_index(['roi_base', 'roi', 'voxel', 'condition']).sort_index()

        mean_df = df.groupby(['roi', 'voxel']).mean(numeric_only=True)
        df = df.join(mean_df[par_labels], on=['roi', 'voxel'], rsuffix='_mean')

        for key in distractor_locations:
            df[f'distance_from_{key}'] =  np.sqrt(
                (df['x'] - distractor_locations[key][0])**2 +
                (df['y'] - distractor_locations[key][1])**2
            )

            df[f'distance_from_{key}_mean'] =  np.sqrt(
                (df['x_mean'] - distractor_locations[key][0])**2 +
                (df['y_mean'] - distractor_locations[key][1])**2
            )

        # Map distractor location to rotation angle (to bring to top)
        rotate_to_up = {loc: (np.pi/2 - angle) for loc, angle in location_angles.items()}

        for key in distractor_locations:
            mask = df.index.get_level_values('condition').str.replace('_', ' ') == key
            df.loc[mask, 'distance_from_distractor'] = df.loc[mask, f'distance_from_{key}']
            df.loc[mask, 'distance_from_distractor_mean'] = df.loc[mask, f'distance_from_{key}_mean']

            df.loc[mask, 'x_rotated'], df.loc[mask, 'y_rotated'] = rotate_x_y(
                df.loc[mask, 'x'], df.loc[mask, 'y'],
                rotate_to_up[key]
            ) 

            df.loc[mask, 'x_mean_rotated'], df.loc[mask, 'y_mean_rotated'] = rotate_x_y(
                df.loc[mask, 'x_mean'], df.loc[mask, 'y_mean'],
                rotate_to_up[key]
            )

        df['distance_from_distractor_rotated'] = np.sqrt((df['x_rotated'] -0)**2 + (df['y_rotated'] - ecc_distractor)**2)
        df['distance_from_distractor_mean_rotated'] = np.sqrt((df['x_mean_rotated'] -0)**2 + (df['y_mean_rotated'] - ecc_distractor)**2)


        for par in par_labels:
            df[f'{par}_diff'] = df[par] - df[f'{par}_mean']

        return df            

ecc_distractor = 4

location_angles = {
    'upper right': np.pi/4,
    'upper left': 3*np.pi/4,
    'lower left': 5*np.pi/4,
    'lower right': 7*np.pi/4
}

distractor_locations = {
    'upper left':  (-ecc_distractor/np.sqrt(2),  ecc_distractor/np.sqrt(2)),
    'upper right': ( ecc_distractor/np.sqrt(2),  ecc_distractor/np.sqrt(2)),
    'lower left':  (-ecc_distractor/np.sqrt(2), -ecc_distractor/np.sqrt(2)),
    'lower right': ( ecc_distractor/np.sqrt(2), -ecc_distractor/np.sqrt(2)),
}




