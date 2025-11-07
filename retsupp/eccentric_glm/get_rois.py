#   size_stimuli: 1.5  # size of the stimuli in visual degrees
#   eccentricity_stimulus: 4.0  # eccentricity of the stimuli in visual degrees

from retsupp.utils.data import Subject
import argparse
from tqdm.contrib.itertools import product
from pathlib import Path
from nilearn import image
import numpy as np
from pathlib import Path
import nibabel as nib


# lh.prf_angle.mgz (subject's LH polar angle, 0-180 degrees refers to UVM -> RHM -> LVM)
# rh.prf_angle.mgz (subject's RH polar angle, 0-180 degrees refers to UVM -> LHM -> RVM)

roi_labels = {('L', 45): 'upper_right',
              ('L', 135): 'lower_right',
              ('R', 45): 'upper_left',
              ('R', 135): 'lower_left'}

def main(subject, mean_prf=True, bids_folder='/data/ds-retsupp'):


    target_dir = Path(bids_folder) / 'derivatives' / 'stimulus_rois' / f'sub-{subject:02d}'
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder=bids_folder)

    # rois = ['V1']
    rois = sub.get_retinotopic_labels().values()

    inferred_pars = sub.get_inferred_pars_volume()

    ecc = inferred_pars['eccen']
    angle = inferred_pars['angle']

    prf_ecc = nib.Nifti1Image(ecc.get_fdata(), affine=ecc.affine)
    prf_angle = nib.Nifti1Image(angle.get_fdata(), affine=angle.affine)

    print(ecc)

    print(angle)


    for hemi in ['L', 'R']:
        for angle in [45, 135]:

            # Get x/y cooordinates for this hemi/angle
            if hemi == 'L':
                x = image.math_img('np.sin(np.deg2rad(angle)) * ecc', angle=prf_angle, ecc=prf_ecc)
                y = image.math_img('np.cos(np.deg2rad(angle)) * ecc', angle=prf_angle, ecc=prf_ecc)

                stim_coord = np.sin(np.deg2rad(angle)) * 4.0, np.cos(np.deg2rad(angle)) * 4.0  # 4.0 is stimulus eccentricity

            else:
                x = image.math_img('-np.sin(np.deg2rad(angle)) * ecc', angle=prf_angle, ecc=prf_ecc)
                y = image.math_img('np.cos(np.deg2rad(angle)) * ecc', angle=prf_angle, ecc=prf_ecc)

                stim_coord = -np.sin(np.deg2rad(angle)) * 4.0, np.cos(np.deg2rad(angle)) * 4.0  # 4.0 is stimulus eccentricity

            sx, sy = stim_coord

            # Compute distance
            dist_to_stim = image.math_img(f'np.sqrt((x - {sx}) ** 2 + (y - {sy}) ** 2)',
                                    x=x, y=y, )

            stim_mask = image.math_img('dist < 1.5', dist=dist_to_stim)  # 1.5 is half size of stimulus

            for roi in rois:
                roi_name = f'{roi}_{hemi}'
                quadrant = roi_labels[(hemi, angle)]

                mask = sub.get_retinotopic_roi(roi_name, bold_space=False)
                mask = nib.Nifti1Image(mask.get_fdata(), affine=mask.affine)

                stim_roi_mask = image.math_img('mask_img.astype(bool) & stim_mask.astype(bool)', mask_img=mask, stim_mask=stim_mask)

                out_fname = target_dir / f'sub-{subject:02d}_desc-{roi_name}_{quadrant}_roi.nii.gz'
                stim_roi_mask.to_filename(out_fname)
                print(f'Saved stimulus ROI to {out_fname}')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int, default=None)
    parser.add_argument('--bids_folder', default='/data/ds-retsupp')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)