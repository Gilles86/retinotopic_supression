rsync_ --include='*/' --include='***/anat/*.shape.gii' --include='***/anat/*.surf.gii' --exclude='*' sciencecluster2:/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep .
sync_ --include='*/' --include='***/*_run-1_*brain_mask.nii.gz' --exclude='*' sciencecluster2:/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep .
rsync_ --include="*/" --include="*.svg" --include="*.html" --exclude="*MNI152" --include="*_desc-preproc_T1w.nii.gz" --exclude="*" sciencecluster2:/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep .
rsync_ \
  --include='*/' \
  --include='***/anat/*.shape.gii' \
  --include='***/anat/*.surf.gii' \
  --include='***/*_run-1_*brain_mask.nii.gz' \
  --include='*.svg' \
  --include='*.html' \
  --include='*_desc-preproc_T1w.nii.gz' \
  --include='***/*_from-fsnative_to-T1w_mode-image_xfm.txt' \
  --include='***/*_space-T1w_boldref.nii.gz' \
  --exclude='*MNI152*' \
  --exclude='*' \
  sciencecluster2:/shares/zne.uzh/gdehol/ds-retsupp/derivatives/fmriprep .
