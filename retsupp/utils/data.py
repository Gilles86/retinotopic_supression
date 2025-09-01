from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from nilearn import image, input_data, surface

class Subject(object):

    def __init__(self, subject_id, bids_folder='/data/ds-retsupp'):
        self.subject_id = int(subject_id)
        self.bids_folder = Path(bids_folder)

    def get_experimental_settings(self, session=1, run=1):
        
        print(self.subject_id, session, run)
        if self.subject_id < 3:
            yml_file = self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id}' / f'ses-{session+1}' / f'sub-{self.subject_id}_ses-{session+1}_task-ret_sup_run-{run}_expsettings.yml'
        else:
            yml_file = self.bids_folder / 'sourcedata' / 'behavior' / 'logs' / f'sub-{self.subject_id:02d}' / f'ses-{session+1}' / f'run-{run}' / f'sub-{self.subject_id:02d}_ses-{session+1}_task-ret_sup_run-{run}_expsettings.yml'

        # Load yml file
        with open(yml_file, 'r') as file:
            settings = yaml.safe_load(file)

        eccentricity_stimuli = settings['experiment'].get('eccentricity_stimulus', 5)
        size_stimuli = settings['experiment'].get('size_stimuli', 1)
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

    def get_stimulus(self, session=1, run=1, resolution=100):

        settings = self.get_experimental_settings(session, run)

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
                pos = -radius_bar_aperture / 2 - bar_width / 2 + dt * speed
            elif current_state in ['bar_left']:
                ori = 0
                pos = radius_bar_aperture / 2 + bar_width / 2 - dt * speed
            elif current_state in ['bar_up']:
                ori = 90
                pos = -radius_bar_aperture / 2 + bar_width / 2 + dt * speed
            elif current_state in ['bar_down']:
                ori = 90
                pos = radius_bar_aperture / 2 - bar_width / 2 - dt * speed

            stimulus[i, ...] = draw_bar(grid_coordinates, pos, ori, bar_width) * mask

        return stimulus

    def get_confounds(self, session=1, run=1, confounds=None):

        def filter_confounds(confounds, n_acompcorr=10):
            confound_cols = ['dvars', 'framewise_displacement']

            a_compcorr_cols = [f"a_comp_cor_{i:02d}" for i in range(n_acompcorr)]
            confound_cols += a_compcorr_cols

            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            motion_cols += [f'{e}_derivative1' for e in motion_cols]
            confound_cols += motion_cols

            steady_state_cols = [c for c in confounds.columns if 'non_steady_state' in c]
            confound_cols += steady_state_cols

            outlier_cols = [c for c in confounds.columns if 'motion_outlier' in c]
            confound_cols += outlier_cols

            cosine_cols = [c for c in confounds.columns if 'cosine' in c]
            confound_cols += cosine_cols

            
            return confounds[confound_cols].fillna(0)

        confounds = pd.read_csv(self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_desc-confounds_timeseries.tsv', sep='\t')

        confounds = filter_confounds(confounds)

        return confounds

    def get_bold(self, session=1, run=1, type='cleaned', return_image=True):

        if type == 'fmriprep':
            fn = self.bids_folder / 'derivatives' / 'fmriprep' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_rec-NORDIC_run-{run}_space-T1w_desc-preproc_bold.nii.gz'
        elif type == 'raw':
            fn = self.bids_folder / 'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_run-{run}_bold.nii.gz'
        elif type == 'cleaned':
            fn = self.bids_folder / 'derivatives' / 'cleaned' / f'sub-{self.subject_id:02d}' / f'ses-{session}' / 'func' / f'sub-{self.subject_id:02d}_ses-{session}_task-search_desc-cleaned_run-{run}_bold.nii.gz'
        else:
            raise ValueError("Type must be 'fmriprep', 'raw', 'cleaned', or 'nordic'")

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
        return labels

    def get_prf_parameters_volume(self, model=1, return_image=False):

        parameters = self.get_prf_parameter_labels(model=model)

        masker = self.get_bold_mask(return_masker=True)

        output = []

        for par in parameters:
            fn = self.bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{self.subject_id:02d}' / f'sub-{self.subject_id:02d}_desc-{par}.nii.gz'
            output.append(pd.Series(masker.fit_transform(fn).squeeze(), name=par))

        output = pd.concat(output, axis=1)

        if return_image:
            return masker.inverse_transform(output.T), parameters
        else:
            return output

    def get_prf_parameters_surface(self, model=1):

        parameters = self.get_prf_parameter_labels(model=model)

        output = []

        for par in parameters:
            tmp = []
            for hemi in ['L', 'R']:
                fn = self.bids_folder / 'derivatives' / 'prf' / f'model{model}' / f'sub-{self.subject_id:02d}' / f'sub-{self.subject_id:02d}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii'
                tmp.append(pd.Series(surface.load_surf_data(fn).squeeze(), name=(par)))
                # output.append(pd.Series(surface.load_surf_data(fn), name=(par, hemi)))
            output.append(pd.concat(tmp, axis=0, keys=['L', 'R'], names=['hemi']))

        output = pd.concat(output, axis=1)

        return output