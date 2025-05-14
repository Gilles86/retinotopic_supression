from  pathlib import Path
from nipype import Node, Workflow
from nipype.interfaces import utility as niu
from nipype.interfaces import ants
from nipype.interfaces import fsl

def create_preprocessed_t1w(subject, session, bids_folder='/data/ds-retsupp',
                            base_dir='/tmp/workflow_folders',
                            t1w_file=None,
                            inv2_file=None,
                            output_filename=None):
    bids_folder = Path(bids_folder)

    if type(subject) == int:
        subject = f'{subject:02d}'

    if output_filename is None:
        output_filename = Path(bids_folder) / f'sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w.nii.gz'

    workflow = Workflow(name=f'bet_workflow_sub-{subject}_session-{session}', base_dir=base_dir)
    input_node = Node(niu.IdentityInterface(fields=['t1w_file', 'inv2_file']),
                    name='input_node')

    if t1w_file is None:
        input_node.inputs.t1w_file = bids_folder / f'sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_T1w.nii.gz'
    else:
        input_node.inputs.t1w_file = t1w_file
    
    if inv2_file is None:
        input_node.inputs.inv2_file = bids_folder / f'sub-{subject}/ses-{session}/anat/sub-{subject}_ses-{session}_inv-2_MP2RAGE.nii.gz'
    else:
        input_node.inputs.inv2_file = inv2_file

    bet = Node(fsl.BET(mask=True),
                name='bet')

    def create_t1w(inv2_file, t1w_file):
        import os
        from nilearn import image
        from nipype.utils.filemanip import filename_to_list, split_filename

        
        # Load the T1w and inv2 images
        t1w_img = image.load_img(t1w_file)
        inv2_img = image.load_img(inv2_file)

        t1w_processed = image.math_img('inv2/inv2.max()*t1w', 
                                        t1w=t1w_img,
                                        inv2=inv2_img)

        # Save the processed T1w image
        # Split the original filename into parts
        _, base, ext = split_filename(t1w_file)

        # Create the new filename with a suffix
        output_file = os.path.abspath(f"{base}_processed{ext}")
        t1w_processed.to_filename(output_file)
        return output_file

    cleaner = Node(niu.Function(input_names=['t1w_file', 'inv2_file'],
                        output_names=['t1w_processed'],
                        function=create_t1w),
                        name='cleaner')


    n4_biasfield = Node(ants.N4BiasFieldCorrection(),
                        name='n4_biasfield')

    n4_biasfield.inputs.output_image = str(output_filename)

    output_node = Node(niu.IdentityInterface(fields=['t1w_processed', 'mask_file']),
                    name='output_node')

    workflow.connect(input_node, 'inv2_file', bet, 'in_file')
    workflow.connect(bet, 'mask_file', n4_biasfield, 'mask_image')
    workflow.connect(cleaner, 't1w_processed', n4_biasfield, 'input_image')
    workflow.connect(input_node, 'inv2_file', cleaner, 'inv2_file')
    workflow.connect(input_node, 't1w_file', cleaner, 't1w_file')
    workflow.connect(n4_biasfield, 'output_image', output_node, 't1w_processed')
    workflow.connect(bet, 'mask_file', output_node, 'mask_file')

    return workflow