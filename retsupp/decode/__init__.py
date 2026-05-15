"""Stimulus decoding from cleaned BOLD via braincoder's ``StimulusFitter``.

We use proper MAP / regularised stimulus inference (NOT a linear-filter
dot-product approximation) on top of fitted m4 (DoG + flex-HRF) PRFs:

1. Fit residual covariance ``omega`` per (subject, ROI) via
   :class:`braincoder.optimize.ResidualFitter`.
2. Decode the per-TR stimulus map with
   :class:`braincoder.optimize.StimulusFitter` against a bar-only
   paradigm from :meth:`Subject.get_bar_stimulus`.

Canonical entrypoint: :mod:`retsupp.decode.decode` (one (subject,
session, run) per call). Calibrated defaults: L2=1.0, lr=0.5,
sd_min=0.2 (passed to the PRF model), resid_max_iter=2000, top-200
voxels by r². See the module docstring + project CLAUDE.md
§"Stimulus decoding entrypoint (canonical)" for the rationale before
changing any of these.
"""
