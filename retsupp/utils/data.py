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

    def get_confounds(self, session=1, run=1, filter_confounds=True):

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
        elif model == 4:
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

    def get_retinotopic_roi(self, roi=None, bold_space=False,):
        """
        Returns a mask image for the specified retinotopic ROI (e.g., 'V1', 'V2', etc.).
        If hemi is 'L' or 'R', restricts to that hemisphere.
        Uses nilearn.image.load_img and image.math_img for masking.
        """
        # Invert the label dictionary for fast lookup

        atlas = self.get_retinotopic_atlas(bold_space=bold_space)
        labels = self.get_retinotopic_labels()

        if roi is None:
            return atlas, [label for _, label in sorted(labels.items())]

        if roi.endswith('_L'):
            hemi = 'L'
            roi = roi[:-2]
        elif roi.endswith('_R'):
            hemi = 'R'
            roi = roi[:-2]
        else:
            hemi = None

        name_to_idx = {v.lower(): k for k, v in labels.items()}
        idx = name_to_idx.get(roi.lower())
        if idx is None:
            raise ValueError(f"ROI '{roi}' not found in label names.")

        roi_mask = image.math_img(f"img == {idx}", img=atlas)

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