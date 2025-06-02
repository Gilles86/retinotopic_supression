function run_nordic(subject_id, run, base_path, session)

if nargin < 3
    base_path = '/shares/zne.uzh/gdehol/ds-retsupp';
end

% Add 'sub-' prefix to subject_id for BIDS compliance
subject = sprintf('sub-%s', subject_id);

% Check if session is provided and format path and filenames accordingly
if nargin < 4 || isempty(session)
    session_str = '';
    session_path = '';  % No session subdirectory
else
    session_str = sprintf('ses-%d_', session);  % Include session in filename
    session_path = sprintf('ses-%d/', session); % Include session in path
end

% Updated input filenames with optional session
fn_magn_in = sprintf('%s/%s/%sfunc/%s_%stask-search_run-%d_part-mag_bold.nii.gz', base_path, subject, session_path, subject, session_str, run);
fn_phase_in = sprintf('%s/%s/%sfunc/%s_%stask-search_run-%d_part-phase_bold.nii.gz', base_path, subject, session_path, subject, session_str, run);

% Updated output directory and filenames with optional session
output_dir = sprintf('%s/derivatives/nordic/%s/%s', base_path, subject, session_path);  % Ensure trailing slash
if ~exist(output_dir, 'dir')
    mkdir(output_dir);  % This will create intermediate directories as needed
end
fn_out = sprintf('sub-%s_%stask-search_run-%d_rec-NORDIC_bold', subject_id, session_str, run);  % New naming convention without .nii
fn_out_full = fullfile(output_dir, [fn_out, '.nii.gz']);

% Debugging: Display the paths being used
disp(['fn_magn_in: ', fn_magn_in]);
disp(['fn_phase_in: ', fn_phase_in]);
disp(['fn_out_full: ', fn_out_full]);

% Set up the ARG structure
ARG.temporal_phase = 1;
ARG.phase_filter_width = 10;
ARG.DIROUT = output_dir;  % Output directory with trailing slash

% Check if the output file already exists and delete it if it does
if exist(fn_out_full, 'file')
    disp(['Deleting existing output file: ', fn_out_full]);
    delete(fn_out_full);
end

% Run NIFTI_NORDIC to denoise the data
disp(['Running NIFTI_NORDIC with output: ', fn_out]);
NIFTI_NORDIC(fn_magn_in, fn_phase_in, fn_out, ARG);

% Verify if the output file is created
if exist(fn_out_full, 'file')
    disp(['Output file created successfully: ', fn_out_full]);
else
    error(['NIFTI_NORDIC did not create the output file: ', fn_out_full]);
end

% Copy the denoised file back to the source folder
final_output = sprintf('%s/%s/%sfunc/%s_%stask-search_task_run-%d_rec-NORDIC_bold.nii.gz', base_path, subject, session_path, subject, session_str, run);
disp(['Copying denoised file to source folder: ', final_output]);
copyfile(fn_out_full, final_output);

end