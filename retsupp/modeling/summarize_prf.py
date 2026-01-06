#!/usr/bin/env python

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker
from retsupp.utils.data import Subject


def main(subject, model, bids_folder="/data/ds-retsupp", prf_type='mean'):
    sub = Subject(subject, bids_folder)
    atlas_img = sub.get_retinotopic_atlas(bold_space=True)
    label_dict = sub.get_retinotopic_labels()
    pars = sub.get_prf_parameters_volume(model=model, type=prf_type)
    dfs = []
    if prf_type == 'runwise':
        for idx, roi_name in label_dict.items():
            roi_mask_img = image.math_img(f"atlas == {idx}", atlas=atlas_img)

            for hemisphere in ['L', 'R']:
                roi_name_ = f"{roi_name}_{hemisphere}"
                hemi_mask = sub.get_hemisphere_mask(hemisphere, bold_space=True)
                mask_ = image.math_img("roi_mask * hemi_mask", roi_mask=roi_mask_img, hemi_mask=hemi_mask)
                print(mask_.affine)
                n_voxels = int(np.sum(mask_.get_fdata() > 0))
                print(f'Processing ROI {roi_name_} ({hemisphere}: {n_voxels} voxels)')
                masker = NiftiMasker(mask_img=mask_)

                pars_img = image.concat_imgs(pars.stack())
                print(f'Parameters image shape: {pars_img.shape}')

                data = masker.fit_transform(pars_img).T
                print(data.shape)
                df_roi = pd.DataFrame(data, columns=pars.stack().index)
                df_roi.index.name = 'voxel'

                df_roi = df_roi.stack(['session', 'run'])

                # Reorder index levels to session, run, voxel
                df_roi = df_roi.reorder_levels(['session', 'run', 'voxel'])
                df_roi.sort_index(inplace=True)

                df_roi = pd.concat((df_roi,), keys=[roi_name_], names=['roi'])

                dfs.append(df_roi)
    elif prf_type == 'conditionwise':

        pars_stacked = pars.stack()

        for idx, roi_name in label_dict.items():
            roi_mask_img = image.math_img(f"atlas == {idx}", atlas=atlas_img)
            for hemisphere in ['L', 'R']:
                roi_name_ = f"{roi_name}_{hemisphere}"
                hemi_mask = sub.get_hemisphere_mask(hemisphere, bold_space=True)
                mask_ = image.math_img("roi_mask * hemi_mask", roi_mask=roi_mask_img, hemi_mask=hemi_mask)
                masker = NiftiMasker(mask_img=mask_)

                pars_img = image.concat_imgs(pars_stacked)

                data = masker.fit_transform(pars_img).T
                print(data.shape)
                df_roi = pd.DataFrame(data, columns=pars_stacked.index).stack('condition')
                df_roi = pd.concat((df_roi,), keys=[roi_name_], names=['roi'])
                df_roi.index = df_roi.index.set_names('voxel', level=1)
                df_roi = df_roi.reorder_levels(['roi', 'condition', 'voxel'])
                df_roi.sort_index(inplace=True)
                print(df_roi)
                dfs.append(df_roi)
    else:
        for idx, roi_name in label_dict.items():
            roi_mask_img = image.math_img(f"atlas == {idx}", atlas=atlas_img)

            for hemisphere in ['L', 'R']:
                roi_name_ = f"{roi_name}_{hemisphere}"
                hemi_mask = sub.get_hemisphere_mask(hemisphere, bold_space=True)
                mask_ = image.math_img("roi_mask * hemi_mask", roi_mask=roi_mask_img, hemi_mask=hemi_mask)
                print(mask_.affine)
                n_voxels = int(np.sum(mask_.get_fdata() > 0))
                print(f'Processing ROI {roi_name_} ({hemisphere}: {n_voxels} voxels)')
                masker = NiftiMasker(mask_img=mask_)

                pars_img = image.concat_imgs([pars[k] for k in pars.keys()])                
                print(f'Parameters image shape: {pars_img.shape}')

                data = masker.fit_transform(pars_img).T
                print(data.shape)
                df_roi = pd.DataFrame(data, columns=pars.index)
                df_roi['roi'] = roi_name_
                df_roi['voxel'] = np.arange(df_roi.shape[0])
                print(df_roi)
                dfs.append(df_roi)

    
    if prf_type == 'runwise':
        df = pd.concat(dfs)
    elif prf_type == 'conditionwise':
        df = pd.concat(dfs)
    else:
        df = pd.concat(dfs, ignore_index=True)
        df.set_index(['roi', 'voxel'], inplace=True)

    if prf_type == 'runwise':
        target_dir = Path(bids_folder) / "derivatives" / "prf_summaries.runwise" / f"model{model}" / f"sub-{subject}"
    elif prf_type == 'conditionwise':
        target_dir = Path(bids_folder) / "derivatives" / "prf_summaries.conditionwise" / f"model{model}" / f"sub-{subject}"
    else:
        target_dir = Path(bids_folder) / "derivatives" / "prf_summaries" / f"model{model}" / f"sub-{subject}"

    target_dir.mkdir(parents=True, exist_ok=True)
    out_csv = target_dir / f"sub-{subject}_model-{model}_prf_voxels.tsv"
    df.to_csv(out_csv, sep='\t')
    print(f"Saved summary to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize voxelwise pRF parameters by retinotopic ROI for a subject.")
    parser.add_argument("subject", type=str, help="Subject ID")
    parser.add_argument("--model", default=1, type=int, help="Model number")
    parser.add_argument("--bids_folder", type=str, default="/data/ds-retsupp", help="BIDS root folder")
    parser.add_argument("--runwise", action="store_true", help="Summarize each run separately.")
    parser.add_argument("--conditionwise", action="store_true", help="Summarize each condition separately.")
    args = parser.parse_args()
    if args.runwise and args.conditionwise:
        raise ValueError("Cannot use --runwise and --conditionwise at the same time.")
    if args.conditionwise:
        prf_type = 'conditionwise'
    else:
        prf_type = 'runwise' if args.runwise else 'mean'
    main(args.subject, args.model, args.bids_folder, prf_type=prf_type)