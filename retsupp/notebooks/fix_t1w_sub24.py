from retsupp.prepare.utils import create_preprocessed_t1w
from nipype.interfaces.fsl import FLIRT
from nilearn import image
import nibabel as nib

# img = image.load_img('/Users/gdehol/data/ds-retsupp/sourcedata/mri/sub-24_ses1/Illuminate_fatwatertestsvivo_WIP_-_WIP_-_acq-MP2_085mm_5_5.nii')

# # img = nib.load(t1w)
# t1w_nii = nib.Nifti1Image(img.dataobj, img.affine)
# t1w_nii = image.index_img(t1w_nii, 0)
# t1w_nii = image.math_img('np.where(img == img.max(), 0, img)', img=t1w_nii)
# print(t1w_nii.get_fdata().min())

# # Rescale the image to 0-4092
# t1w_nii = image.math_img('img - img.min()', img=t1w_nii)

# print(t1w_nii.get_fdata().min())
# t1w_nii.to_filename('/data/ds-retsupp/sub-24/ses-1/anat/sub-24_ses-1_T1w_thijs.nii.gz')


# flirt = FLIRT()

# flirt.inputs.in_file = '/Users/gdehol/data/ds-retsupp/sub-24/ses-1/anat/sub-24_ses-1_T1w_thijs.nii.gz'
# flirt.inputs.reference = '/data/ds-retsupp/sub-24/ses-1/anat/sub-24_ses-1_inv-1_MP2RAGE.nii.gz'
# flirt.inputs.dof = 6

# flirt.inputs.in_matrix_file = '/data/ds-retsupp/sub-24/ses-1/anat/init.mat'

# flirt.inputs.out_file = '/Users/gdehol/data/ds-retsupp/sub-24/ses-1/anat/sub-24_ses-1_T1w.nii.gz'
# flirt.inputs.searchr_x = [-5, 5]
# flirt.inputs.searchr_y = [-5, 5]
# flirt.inputs.searchr_z = [-5, 5]
# flirt.inputs.cost = 'mutualinfo'
# flirt.inputs.interp = 'nearestneighbour'

# res = flirt.run()


wf = create_preprocessed_t1w(24, 1, '/data/ds-retsupp')
wf.run()