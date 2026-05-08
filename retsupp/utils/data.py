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

    def get_distractor_mapping_old(self):
        subject = int(self.subject_id)

        # Counterbalancing lists (fixed indexing: (subject-1) % 8)
        hp_sequences = [
            [1, 5, 3, 7],
            [1, 5, 7, 3],
            [3, 7, 5, 1],
            [5, 1, 7, 3],
            [3, 7, 1, 5],
            [7, 3, 1, 5],
            [5, 1, 3, 7],
            [7, 3, 5, 1],
        ]
        hp_list = hp_sequences[(subject - 1) % 8]

        # Build the 12-run block pattern (2 sessions × 6 runs)
        if subject in (1, 2):
            # Bugged order: AA BBB CCC DDD A  →  runs: 1..12 = A,A,B,B,B,C,C,C,D,D,D,A
            block_order = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
        else:
            # Intended order: AAA BBB CCC DDD  →  runs: 1..12 = A,A,A,B,B,B,C,C,C,D,D,D
            block_order = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

        distractor_locations = {}
        for session in (1, 2):
            for run in range(1, 7):
                global_run = (session - 1) * 6 + run  # 1..12
                block_idx = block_order[global_run - 1]  # 0..3
                distractor_locations[(session, run)] = self.location_mapping[hp_list[block_idx]]

        return distractor_locations

    def get_experimental_settings(self, session=1, run=1):
        
        print(self.subject_id, session, run)
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
        if (self.subject_id == 20) & (session == 1):
            return [1,2,3,4,5]
        elif (self.subject_id == 24) & (session == 2):
            return [1,2,3,4,5]

        if (self.subject_id == 24) & (session == 2):
            return [1,2,3,4, 5]

        return [1,2,3,4,5,6]

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
        distractor_long_side=1.5, distractor_short_side=0.5,
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

        # --- Distractor stimulus pass. ---
        # For each target event with a real distractor (locations 1/3/5/7),
        # paint a disk at the distractor location with intensity equal to
        # the fraction of the TR during which the distractor was on.
        targets = onsets[onsets["event_type"] == "target"].sort_values("onset")
        feedback = onsets[onsets["event_type"] == "feedback"].sort_values("onset")
        # Map distractor location code → (x, y) on the ring.
        # 1: upper_right, 3: upper_left, 5: lower_left, 7: lower_right.
        loc_xy = {
            1.0: (4 / np.sqrt(2),  4 / np.sqrt(2)),
            3.0: (-4 / np.sqrt(2),  4 / np.sqrt(2)),
            5.0: (-4 / np.sqrt(2), -4 / np.sqrt(2)),
            7.0: (4 / np.sqrt(2), -4 / np.sqrt(2)),
        }
        # Pre-compute TR window edges for fast overlap math.
        tr_starts = frametimes - tr / 2.
        tr_ends = frametimes + tr / 2.

        for _, trial in targets.iterrows():
            loc_code = trial["distractor_location"]
            if pd.isna(loc_code) or loc_code == 10.0:
                continue
            cx, cy = loc_xy[loc_code]
            t_on = trial["onset"]
            after = feedback[feedback["onset"] > t_on]
            t_off = after.iloc[0]["onset"] if len(after) else t_on + max_distractor_duration
            # Cap.
            t_off = min(t_off, t_on + max_distractor_duration)

            # Fraction of TR overlapping [t_on, t_off].
            overlap = np.clip(
                np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
                0.0, None,
            ) / tr  # in [0, 1]
            if distractor_shape == 'circle':
                footprint = (((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <
                             distractor_radius ** 2).astype(np.float32)
            else:
                # Rectangle: rotate pixel grid into the rectangle's local
                # frame and binary inside-test on |xr| <= long/2,
                # |yr| <= short/2. ``distractor_orientation`` is in
                # degrees; 0° = long axis horizontal, 90° = vertical.
                ori_deg = trial.get('distractor_orientation', np.nan)
                if pd.isna(ori_deg):
                    # Fall back to circle if we somehow lack the
                    # orientation (e.g. older events.tsv without
                    # target_orientation). Should not happen in practice.
                    footprint = (
                        ((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                        < distractor_radius ** 2
                    ).astype(np.float32)
                else:
                    ang = -float(ori_deg) * np.pi / 180.0
                    cos_a = np.cos(ang)
                    sin_a = np.sin(ang)
                    dx = grid_x - cx
                    dy = grid_y - cy
                    xr = dx * cos_a + dy * sin_a
                    yr = -dx * sin_a + dy * cos_a
                    footprint = (
                        (np.abs(xr) <= distractor_long_side / 2.0) &
                        (np.abs(yr) <= distractor_short_side / 2.0)
                    ).astype(np.float32)

            # Where this distractor is brighter than what's already there,
            # update.  This handles overlapping distractor windows
            # (rare, but keeps it idempotent).
            active = overlap > 0
            if active.any():
                contrib = overlap[active, None, None] * footprint[None, :, :]
                # Element-wise max with current stimulus over those frames.
                stimulus[active] = np.maximum(stimulus[active], contrib)

            if debug:
                ori_dbg = trial.get('distractor_orientation', np.nan)
                print(
                    f"[DEBUG] trial @ t={t_on:.2f}s, loc={loc_code}, "
                    f"shape={distractor_shape}, ori={ori_dbg}, "
                    f"on=[{t_on:.2f}, {t_off:.2f}], "
                    f"max overlap fraction={overlap.max():.3f}"
                )

        return stimulus

    def get_dynamic_indicator(self, session=1, run=1,
                              max_distractor_duration=1.5,
                              oversampling=1):
        """Per-TR distractor-on indicator at each of the 4 ring locations.

        Parameters
        ----------
        oversampling : int, default 1
            Temporal oversampling factor. When ``> 1``, the indicator is
            computed on a fine time grid with step ``dt = tr /
            oversampling`` instead of TR. Each output row then represents
            the on-fraction of one fine sub-bin (length ``dt``), not of a
            full TR. Returned shape becomes ``(n_volumes * oversampling,
            4)``. With ``oversampling=1`` the result is exactly the
            previous behaviour.

        Returns
        -------
        d : ndarray, shape (n_volumes * oversampling, 4)
            Fraction of each (sub-)bin during which a distractor was on
            screen at each ring location, in [0, 1]. The 4 channels are
            ordered as
            ``['upper_right', 'upper_left', 'lower_left', 'lower_right']``
            to match the ``CONDITIONS`` list used by
            :mod:`retsupp.modeling.fit_af_prf_braincoder` /
            :mod:`retsupp.modeling.fit_dynamic_af_braincoder` (and the
            ``ring_positions`` ordering passed to the
            ``DynamicAttentionFieldPRF2DWithHRF`` model).

        Notes
        -----
        Logic mirrors the distractor pass of
        :meth:`get_stimulus_with_distractors`, but does not paint any
        spatial grid — it only returns the per-bin per-location overlap
        fraction.
        """
        if int(oversampling) < 1:
            raise ValueError(f"oversampling must be >= 1, got {oversampling}")
        oversampling = int(oversampling)

        # Ring location code -> channel index. Channel order MUST stay
        # in sync with `CONDITIONS` in fit_*_af_braincoder.py.
        # 1: upper_right, 3: upper_left, 5: lower_left, 7: lower_right.
        loc_to_channel = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}
        n_channels = 4

        tr = self.get_tr(session, run)
        n_volumes = self.get_n_volumes(session, run)
        dt = tr / oversampling
        n_bins = n_volumes * oversampling
        # Bin centres at dt/2, 3*dt/2, ... ; bin edges = centre ± dt/2.
        frametimes = (np.arange(n_bins, dtype=np.float64) + 0.5) * dt
        tr_starts = frametimes - dt / 2.
        tr_ends = frametimes + dt / 2.

        onsets = self.get_onsets(session, run)
        targets = onsets[onsets["event_type"] == "target"].sort_values("onset")
        feedback = onsets[onsets["event_type"] == "feedback"].sort_values("onset")

        d = np.zeros((len(frametimes), n_channels), dtype=np.float32)

        for _, trial in targets.iterrows():
            loc_code = trial["distractor_location"]
            if pd.isna(loc_code) or loc_code == 10.0:
                continue
            if loc_code not in loc_to_channel:
                continue
            ch = loc_to_channel[loc_code]

            t_on = trial["onset"]
            after = feedback[feedback["onset"] > t_on]
            t_off = (after.iloc[0]["onset"] if len(after)
                     else t_on + max_distractor_duration)
            t_off = min(t_off, t_on + max_distractor_duration)

            overlap = np.clip(
                np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
                0.0, None,
            ) / dt  # in [0, 1]

            # Element-wise max in case of overlapping windows.
            d[:, ch] = np.maximum(d[:, ch], overlap.astype(np.float32))

        return d

    def get_target_indicator(self, session=1, run=1,
                             max_target_duration=1.5,
                             oversampling=1):
        """Per-TR target-on indicator at each of the 4 ring locations.

        Identical machinery to :meth:`get_dynamic_indicator`, but reads
        the ``target_location`` column instead of ``distractor_location``
        — i.e. tracks the per-TR overlap fraction of the SEARCH TARGET
        (not the singleton distractor) at each of the 4 ring positions.

        Used by the v3 + target model
        (``DoGDynamicAttentionFieldPRF2DWithHRF_v3_target``) as a
        positive-control "phasic capture" channel: the same visual
        transient that suppresses at the distractor should produce
        positive gain at the target.

        Parameters
        ----------
        oversampling : int, default 1
            Temporal oversampling factor. See
            :meth:`get_dynamic_indicator` for details.

        Returns
        -------
        t : ndarray, shape (n_volumes * oversampling, 4)
            Fraction of each (sub-)bin during which a target was on
            screen at each ring location, in [0, 1]. Channel order is
            ``['upper_right', 'upper_left', 'lower_left', 'lower_right']``
            (same as ``get_dynamic_indicator``).
        """
        if int(oversampling) < 1:
            raise ValueError(f"oversampling must be >= 1, got {oversampling}")
        oversampling = int(oversampling)

        # Same channel mapping as get_dynamic_indicator.
        loc_to_channel = {1.0: 0, 3.0: 1, 5.0: 2, 7.0: 3}
        n_channels = 4

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

        t = np.zeros((len(frametimes), n_channels), dtype=np.float32)

        for _, trial in targets.iterrows():
            loc_code = trial["target_location"]
            if pd.isna(loc_code) or loc_code == 10.0:
                continue
            if loc_code not in loc_to_channel:
                continue
            ch = loc_to_channel[loc_code]

            t_on = trial["onset"]
            after = feedback[feedback["onset"] > t_on]
            t_off = (after.iloc[0]["onset"] if len(after)
                     else t_on + max_target_duration)
            t_off = min(t_off, t_on + max_target_duration)

            overlap = np.clip(
                np.minimum(tr_ends, t_off) - np.maximum(tr_starts, t_on),
                0.0, None,
            ) / dt  # in [0, 1]

            # Element-wise max in case of overlapping windows.
            t[:, ch] = np.maximum(t[:, ch], overlap.astype(np.float32))

        return t

        def filter_confounds_(confounds, n_acompcorr=10):
            confound_cols = ['dvars', 'framewise_displacement']

            # Only include available a_comp_cor columns, up to n_acompcorr
            a_compcorr_cols = [f'a_comp_cor_{i:02d}' for i in range(n_acompcorr)]
            a_compcorr_cols = [c for c in a_compcorr_cols if c in confounds.columns]
            confound_cols += a_compcorr_cols

            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            motion_cols += [f'{e}_derivative1' for e in motion_cols]
            confound_cols += [c for c in motion_cols if c in confounds.columns]

            steady_state_cols = [c for c in confounds.columns if 'non_steady_state' in c]
            confound_cols += steady_state_cols

            outlier_cols = [c for c in confounds.columns if 'motion_outlier' in c]
            confound_cols += outlier_cols

            cosine_cols = [c for c in confounds.columns if 'cosine' in c]
            confound_cols += cosine_cols

            # Only keep columns that exist in confounds
            confound_cols = [c for c in confound_cols if c in confounds.columns]
            return confounds[confound_cols].fillna(0)

        confounds = pd.read_csv(self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_desc-confounds_timeseries.tsv', sep='\t')

        if filter_confounds:
            confounds = filter_confounds_(confounds)

        return confounds

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

    def get_bold_mask(self, session=1, run=1, return_masker=False):
        fn = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_space-T1w_desc-brain_mask.nii.gz'
        
        if return_masker:
            return input_data.NiftiMasker(mask_img=fn)
        else:
            return image.load_img(fn)

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
            else:
                masker = self.get_bold_mask(return_masker=True)
                data = {par: self._extract_param_arr(images[par], roi=None) for par in param_labels}
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
            # Four conditions: upper_left, upper_right, lower_left, lower_right
            conditions = ['upper_left', 'upper_right', 'lower_left', 'lower_right']
            base_dir = self.bids_folder / 'derivatives' / 'prf_conditionfit' / f'model{model}' / f'sub-{self.subject_id:02d}'
            data = []
            index = []
            for cond in conditions:
                row = {}
                for par in param_labels:
                    img_path = base_dir / f'sub-{self.subject_id:02d}_cond-{cond}_desc-{par}.nii.gz'
                    row[par] = nib.load(str(img_path))
                data.append(row)
                index.append(cond)
            df = pd.DataFrame(data, index=pd.Index(index, name='condition'))
            df.columns.name = 'parameter'
            return df
        else:
            raise ValueError("type must be 'mean', 'runwise', or 'conditionwise'")

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


    def get_retinotopic_atlas(self, bold_space=False):
        varea_file = (
            self.bids_folder
            / "derivatives"
            / "fmriprep"
            / "sourcedata"
            / "freesurfer"
            / f"sub-{self.subject_id:02d}"
            / "mri"
            / "inferred_varea.mgz"
        )
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

    def get_retinotopic_roi(self, roi=None, bold_space=False,):
        """
        Returns a mask image for the specified retinotopic ROI (e.g., 'V1', 'V2', etc.).
        If hemi is 'L' or 'R', restricts to that hemisphere.

        Sources the atlas based on the ROI name:
        - Benson (``inferred_varea.mgz``) for V1, V2, V3, hV4, VO1, VO2,
          LO1, LO2, TO1, TO2, V3A, V3B, plus the alias ``V3AB`` (=V3A∪V3B).
        - Wang 2015 (``wang15_atlas.mgz``) for IPS0..IPS5, SPL1, FEF,
          plus the alias ``IPS`` (=IPS0∪..∪IPS5).

        ROI names ending in ``_L`` or ``_R`` are restricted to the
        corresponding hemisphere via :meth:`get_hemisphere_mask`.
        """
        if roi is None:
            atlas = self.get_retinotopic_atlas(bold_space=bold_space)
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
            name_to_idx = {v.lower(): k for k, v in labels.items()}
            if roi in self._WANG_ALIASES:
                component_rois = self._WANG_ALIASES[roi]
            else:
                component_rois = [roi]
            indices = []
            for r in component_rois:
                idx = name_to_idx.get(r.lower())
                if idx is None:
                    raise ValueError(f"ROI '{r}' not found in Wang label names.")
                indices.append(idx)
            cond = ' | '.join(f'(img == {i})' for i in indices)
            roi_mask = image.math_img(cond, img=atlas)
        else:
            atlas = self.get_retinotopic_atlas(bold_space=bold_space)
            labels = self.get_retinotopic_labels()
            name_to_idx = {v.lower(): k for k, v in labels.items()}
            if roi in self._BENSON_ALIASES:
                component_rois = self._BENSON_ALIASES[roi]
            else:
                component_rois = [roi]
            indices = []
            for r in component_rois:
                idx = name_to_idx.get(r.lower())
                if idx is None:
                    raise ValueError(f"ROI '{r}' not found in label names.")
                indices.append(idx)
            cond = ' | '.join(f'(img == {i})' for i in indices)
            roi_mask = image.math_img(cond, img=atlas)

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


    def get_inferred_pars_volume(self, return_images=True):
        from nibabel.freesurfer.io import read_morph_data
        import numpy as np
        import pandas as pd
        from itertools import product

        freesurfer_dir = self.bids_folder / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer' / f'sub-{self.subject_id:02d}' / 'mri'
        par_labels = ['angle', 'eccen', 'sigma', 'varea']

        results = {}

        for par in par_labels:
            img = nib.load(freesurfer_dir / f'inferred_{par}.mgz')
            results[par] = img

        if return_images:
            return pd.Series(results)
        else:
            masker = self.get_bold_mask(return_masker=True)
            data = {par: self._extract_param_arr(results[par], roi=None) for par in par_labels}
            return pd.DataFrame(data)

    def get_inferred_prf_pars_surf(self):
        from nibabel.freesurfer.io import read_morph_data
        import numpy as np
        import pandas as pd
        from itertools import product

        freesurfer_dir = self.bids_folder / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer' / f'sub-{self.subject_id:02d}'
        par_labels = ['angle', 'eccen', 'sigma', 'varea']
        hemis = ['L', 'R']
        fs_hemi = {'L': 'lh', 'R': 'rh'}

        # Load data for each hemisphere
        dfs = []
        for hemi in hemis:
            df = pd.DataFrame({
                par: read_morph_data(freesurfer_dir / 'surf' / f'{fs_hemi[hemi]}.inferred_{par}').squeeze().astype(np.float32)
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




