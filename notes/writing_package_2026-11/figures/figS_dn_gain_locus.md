**Figure S?. Where in the divisive-normalization (DN) pRF model does the sustained HP effect act?** a, DN response
of a voxel (Aqil et al., 2021) with the three places a sustained field can act, each one field per subject × ROI
(shared parameters, fitted on all voxels of the ROI; each voxel sees the field over its own pRF): the stimulus drive,
in numerator and denominator (input gain, the model of Fig. 5, blue); the neural baseline b (teal); the surround
baseline d (purple). The b- and d-fields share one shape (width σ_AF, HP/LP pattern θ) with separate weights w_b, w_d;
at w = 0 each equals the input-gain model exactly. Scaling b and d together equals scaling the amplitude a, so only
w_b − w_d (b/d) is specific to DN. b, Top: held-out R² of each variant minus the input-gain model, mean ± SEM over
subjects (run3 folds; per-voxel pRFs frozen at the all-runs fit in every model, shared terms refitted per fold).
Bottom: dS = g_HP − g_LP of the input gain in each model (all-runs fits). c, Recovery on pending: BOLD simulated
from an input, b or d effect of equal BOLD size plus phase-randomised real residuals, refitted with every model.
Confusion: % of simulations in which each model is selected (training-SSE gain of the b- or d-field over the input
model, threshold = 95th percentile under the input truth, so the input row is ~95 % by construction). Right: dS
fitted by the input-gain model under each truth, relative to the input truth's dS. d, Predicted change of the
neural response in voxels of the HP quadrant, the two next to it and the opposite one (median over synthetic voxels
on the diagonals at 1–3°, DN parameters in the range of the fits; ``sim_dn_field_signatures``), for an HP field on
each locus and, for comparison, the DoG with the same input field; bar over the pRF centre or in its surround. Each
location type minus the mean of the four quadrants, scaled and signed so that HP = −1 (only the shape is compared).
Right: measured, same scaling: item responses in early areas (Fig. 3e) and the decoded bar (DN pRFs, per-ROI HRF;
Fig. 6 with DN), mean ± SEM, n = 30. e,
Model-free: per-HP-condition DN fits (n = 30, no CV), HP-quadrant minus opposite-quadrant voxels; rings,
p < .05 BH over ROIs.
