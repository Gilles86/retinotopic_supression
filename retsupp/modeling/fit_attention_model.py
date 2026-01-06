#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

# Distractor coordinate mapping
def get_distractor_coords(label, radius=4.):
    mapping = {
        'upper_right': np.pi/4,
        'upper_left': 3*np.pi/4,
        'lower_left': 5*np.pi/4,
        'lower_right': 7*np.pi/4
    }
    angle = mapping.get(label, None)
    if angle is None:
        return None, None
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y

def get_conditionwise_pars(subject, bids_folder, r2_thr=0.1, ecc_thr=3.0):
    """Load and preprocess condition-wise PRF parameters"""
    par_labels = ['x', 'y', 'sd', 'amplitude', 'ecc', 'r2', 'srf_size', 'srf_amplitude', 'hrf_delay', 'hrf_dispersion']
    subject = int(subject)
    # Load data
    condition_pars = pd.read_csv(
        bids_folder / f'derivatives/prf_summaries.conditionwise/model4/sub-{subject:02d}/sub-{subject:02d}_model-4_prf_voxels.tsv',
        sep='\t', index_col=[0, 1, 2]
    )
    # Calculate mean parameters
    condition_mean_pars = condition_pars.groupby(['roi', 'voxel']).mean()
    condition_pars = condition_pars.join(condition_mean_pars[par_labels], rsuffix='_mean')
    # Clean up and add difference columns
    condition_pars.reset_index(inplace=True)
    condition_pars['roi'] = condition_pars['roi'].str.replace('_L', '').str.replace('_R', '')
    condition_pars.set_index(['roi', 'voxel', 'condition'], inplace=True)
    for par in par_labels:
        condition_pars[par + '_diff'] = condition_pars[par] - condition_pars[par + '_mean']
    condition_pars = condition_pars[condition_pars['r2'] > r2_thr]
    condition_pars = condition_pars[condition_pars['ecc_mean'] < ecc_thr]
    condition_pars['empirical_dx'] = condition_pars['x'] - condition_pars['x_mean']
    condition_pars['empirical_dy'] = condition_pars['y'] - condition_pars['y_mean']
    return condition_pars

def compute_net_shifts(prf_x, prf_y, prf_sd, current_condition, attention_sd, ratio):
    """
    Net shift from 4 attention Gaussians multiplied with a PRF Gaussian.
    Uses the exact precision-weighted mean for the product of Gaussians.
    """
    distractor_names = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
    coords = {name: get_distractor_coords(name) for name in distractor_names}
    # Variances and precisions
    var_prf = prf_sd**2
    prec_prf = 1.0 / var_prf
    # Build per-distractor variances/precisions (attended vs others)
    var_att = []
    mu_x = []
    mu_y = []
    for name in distractor_names:
        att_sd = (attention_sd * ratio) if name == current_condition else attention_sd
        var_att.append(att_sd**2)
        x_i, y_i = coords[name]
        mu_x.append(x_i)
        mu_y.append(y_i)
    prec_att = [1.0 / v for v in var_att]
    # Product of Gaussians: posterior mean = (Σ precisions * means) / (Σ precisions)
    denom = prec_prf + sum(prec_att)
    post_mu_x = (prec_prf * prf_x + sum(p * m for p, m in zip(prec_att, mu_x))) / denom
    post_mu_y = (prec_prf * prf_y + sum(p * m for p, m in zip(prec_att, mu_y))) / denom
    # Net shift relative to original PRF center
    net_dx = post_mu_x - prf_x
    net_dy = post_mu_y - prf_y
    return net_dx, net_dy

def compute_error(df, attention_sd, ratio):
    total_error = 0.0
    for condition in ['upper_right', 'upper_left', 'lower_left', 'lower_right']:
        if condition in df['condition'].unique():
            group = df[df['condition'] == condition]
            for _, row in group.iterrows():
                dx, dy = compute_net_shifts(row['x_mean'], row['y_mean'], row['sd_mean'],
                                            condition, attention_sd, ratio)
                total_error += np.sqrt((dx - row['empirical_dx'])**2 + (dy - row['empirical_dy'])**2)
    return total_error

def compute_condition_shifts(df, current_condition, attention_sd, ratio):
    """Compute all shifts for one condition"""
    distractor_coords = {name: get_distractor_coords(name) for name in ['upper_right', 'upper_left', 'lower_left', 'lower_right']}
    attention_x, attention_y = distractor_coords[current_condition]
    group = df[df['condition'] == current_condition].copy()
    group['empirical_dx'] = group['x'] - group['x_mean']
    group['empirical_dy'] = group['y'] - group['y_mean']
    group['empirical_shift'] = np.sqrt(group['empirical_dx']**2 + group['empirical_dy']**2)
    group['original_dist'] = np.sqrt((group['x_mean'] - attention_x)**2 + (group['y_mean'] - attention_y)**2)
    group['new_dist'] = np.sqrt((group['x'] - attention_x)**2 + (group['y'] - attention_y)**2)
    group['empirical_shift_relative'] = group['new_dist'] - group['original_dist']
    group['predicted_dx'], group['predicted_dy'] = zip(*[
        compute_net_shifts(x, y, sd, current_condition, attention_sd, ratio)
        for x, y, sd in zip(group['x_mean'], group['y_mean'], group['sd_mean'])
    ])
    group['predicted_shift'] = np.sqrt(group['predicted_dx']**2 + group['predicted_dy']**2)
    group['predicted_new_x'] = group['x_mean'] + group['predicted_dx']
    group['predicted_new_y'] = group['y_mean'] + group['predicted_dy']
    group['predicted_new_dist'] = np.sqrt(
        (group['predicted_new_x'] - attention_x)**2 + (group['predicted_new_y'] - attention_y)**2
    )
    group['predicted_shift_relative'] = group['predicted_new_dist'] - group['original_dist']
    group['attention_sd'] = attention_sd
    group['ratio'] = ratio
    group['main_attention_x'] = attention_x
    group['main_attention_y'] = attention_y
    return group

def fit_attention_params(df):
    def error_function(params):
        attention_sd, log_ratio = params
        ratio = np.exp(log_ratio)
        return compute_error(df, attention_sd, ratio)

    initial_guess = [10.0, 0.0]  # attention_sd=10.0, log_ratio=0.0
    bounds = [(10, 100), (-5.5, 5.5)]  # attention_sd >= 1, log_ratio bounds
    
    def callback(xk):
        print(f"Current params: attention_sd={xk[0]:.3f}, log_ratio={xk[1]:.3f}, error={error_function(xk):.3f}")

    result = minimize(
        error_function,
        initial_guess,
        bounds=bounds,
        method='Powell',
        options={'maxiter': 1000, 'ftol': 1e-6, 'xtol': 1e-6, 'disp': True},
        callback=callback
)
    return result.x[0], result.x[1], result.success, result.fun

def create_hexbin_plots(final_df, subject, roi, output_pdf, attention_sd, ratio):
    """Create hexbin plots and save to PDF"""
    distractor_names = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
    distractor_coords = {name: get_distractor_coords(name) for name in distractor_names}
    # Find global min/max for consistent color scaling
    all_shifts = np.concatenate([
        final_df['empirical_shift_relative'].values,
        final_df['predicted_shift_relative'].values
    ])
    # global_vmax = max(abs(all_shifts.min()), abs(all_shifts.max()))
    global_vmax = 1.
    norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0, vmax=global_vmax)
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(4, 2, figsize=(16, 24))
        fig.suptitle(
            f'Subject {subject} - {roi} - Shifts Relative to Main Distractor\n'
            f'(Attention SD: {attention_sd:.2f}, Distractor ratio: {np.exp(ratio):.2f})',
            y=1.02
        )
        for i, condition in enumerate(distractor_names):
            if condition in final_df['condition'].unique():
                condition_data = final_df[final_df['condition'] == condition]
                attention_x, attention_y = distractor_coords[condition]
                attention_x *= 1.2
                attention_y *= 1.2
                # Empirical shifts plot
                ax = axes[i, 0]
                hb = ax.hexbin(condition_data['x_mean'], condition_data['y_mean'],
                             C=condition_data['empirical_shift_relative'],
                             gridsize=10, cmap='coolwarm', norm=norm, mincnt=1)
                ax.scatter(attention_x, attention_y, color='black', s=200, marker='*')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'{condition} - Empirical Shifts')
                ax.set_xlim(-4., 4.)
                ax.set_ylim(-4., 4.)
                fig.colorbar(hb, ax=ax, label='Relative Shift (red=away, blue=toward)')
                # Predicted shifts plot
                ax = axes[i, 1]
                hb = ax.hexbin(condition_data['x_mean'], condition_data['y_mean'],
                             C=condition_data['predicted_shift_relative'],
                             gridsize=10, cmap='coolwarm', norm=norm, mincnt=1)
                ax.scatter(attention_x, attention_y, color='black', s=200, marker='*')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'{condition} - Predicted Shifts')
                ax.set_xlim(-4., 4.)
                ax.set_ylim(-4., 4.)
                fig.colorbar(hb, ax=ax, label='Relative Shift (red=away, blue=toward)')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        # Add scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        for condition in distractor_names:
            if condition in final_df['condition'].unique():
                condition_data = final_df[final_df['condition'] == condition]
                ax1.scatter(condition_data['empirical_shift'], condition_data['predicted_shift'],
                          alpha=0.5, label=condition)
                ax2.scatter(condition_data['empirical_shift_relative'], condition_data['predicted_shift_relative'],
                          alpha=0.5, label=condition)
        ax1.plot([0, 3], [0, 3], 'r--')
        ax1.set_xlabel('Empirical Total Shift')
        ax1.set_ylabel('Predicted Total Shift')
        ax1.set_title('Total Shifts')
        ax1.legend()
        ax1.grid(True)
        ax2.plot([-2, 2], [-2, 2], 'r--')
        ax2.set_xlabel('Empirical Relative Shift (positive = away)')
        ax2.set_ylabel('Predicted Relative Shift (positive = away)')
        ax2.set_title('Shifts Relative to Main Distractor')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

def process_subject(subject_id, bids_folder):
    """Process a single subject"""
    # Create output directory
    output_dir = bids_folder / f'derivatives/attention_model/sub-{subject_id:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load data
    condition_pars = get_conditionwise_pars(subject_id, bids_folder)
    model_pars = []
    # Process each ROI
    results = []
    for roi, roi_data in condition_pars.groupby(level='roi'):
        print(f"Processing ROI: {roi}")
        # Prepare data for this ROI
        test_pars = roi_data.reset_index()
        test_pars = test_pars[['x', 'y', 'sd', 'x_mean', 'y_mean', 'sd_mean', 'condition', 'voxel', 'empirical_dx', 'empirical_dy']]
        # Filter to only include our conditions
        distractor_names = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
        test_pars = test_pars[test_pars['condition'].isin(distractor_names)]
        if len(test_pars) == 0:
            print(f"No data for ROI {roi}, skipping...")
            continue
        # Fit model
        best_attention_sd, best_log_ratio, success, final_error = fit_attention_params(test_pars)
        best_ratio = np.exp(best_log_ratio)
        print(f"ROI: {roi}")
        print(f"Fitted attention_sd: {best_attention_sd:.3f}")
        print(f"Fitted ratio (log): {best_log_ratio:.3f}, value: {best_ratio:.3f}")
        print(f"Final error: {final_error:.2f}")
        model_pars.append({
            'subject': f'{subject_id:02d}',
            'roi': roi,
            'fitted_attention_sd': best_attention_sd,
            'fitted_ratio_log': best_log_ratio,
            'fitted_ratio': best_ratio,
            'final_error': final_error,
            'optimization_success': success
        })
        # Compute results for all conditions
        roi_results = []
        for condition in distractor_names:
            if condition in test_pars['condition'].unique():
                condition_results = compute_condition_shifts(
                    test_pars, condition, best_attention_sd, best_ratio
                )
                roi_results.append(condition_results)
        if roi_results:
            final_df = pd.concat(roi_results)
            final_df['roi'] = roi
            final_df['subject'] = subject_id
            results.append(final_df)
            # Save plots
            output_pdf = output_dir / f'sub-{subject_id:02d}_roi-{roi}_desc-attentionmodel_plots.pdf'
            create_hexbin_plots(final_df, subject_id, roi, output_pdf, best_attention_sd, best_log_ratio)
    # Save all results to TSV
    if results:
        all_results = pd.concat(results)
        output_tsv = output_dir / f'sub-{subject_id:02d}_desc-attentionmodel_pars.tsv'
        all_results.to_csv(output_tsv, sep='\t', index=False)
        print(f"Saved results to {output_tsv}")
        model_pars = pd.DataFrame(model_pars)
        output_model_pars_tsv = output_dir / f'sub-{subject_id:02d}_desc-attentionmodel_summary.tsv'
        model_pars.to_csv(output_model_pars_tsv, sep='\t', index=False)
        print(f"Saved model summary to {output_model_pars_tsv}")
    else:
        print("No results to save.")

if __name__ == '__main__':
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fit attention model to PRF data')
    parser.add_argument('subject_id', type=int, help='Subject ID (e.g., "01")')
    parser.add_argument('--bids_folder', default='/data/ds-retsupp', type=Path, help='Path to BIDS folder')
    # Parse arguments
    args = parser.parse_args()
    # Run the analysis
    process_subject(args.subject_id, args.bids_folder)
