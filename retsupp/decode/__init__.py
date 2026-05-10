"""Stimulus decoding from cleaned BOLD via braincoder's ``StimulusFitter``.

We use proper MAP / regularised stimulus inference (NOT a linear-filter
dot-product approximation) on top of fitted model-4 DoG PRFs:

1. Fit residual covariance ``omega`` per (subject, ROI) via
   :class:`braincoder.optimize.ResidualFitter`.
2. Decode the per-TR stimulus map with
   :class:`braincoder.optimize.StimulusFitter`
   (``l2_norm=0.01``, ``learning_rate=0.01``, ``max_n_iterations=1000``).
3. Sample the decoded image inside a small disk at each of the four
   ring positions (4 deg eccentricity, on the diagonals).
4. Per (run, ring), classify the position as HP / orth / opposite
   relative to that run's HP-distractor location and aggregate.

The pipeline mirrors lesson 7 of the braincoder tutorial
(``examples/00_encodingdecoding/fit_prf.py``), specialised for retsupp.

Entry points
------------
* :mod:`retsupp.decode.decoder` -- shared core (``decode_run``).
* :mod:`retsupp.decode.smoke_test` -- one-subject sanity check
  (sub-02 V1, one run) that saves a spatial-map figure.
* :mod:`retsupp.decode.run_decode` -- per-(subject, ROI) batch driver.
* :mod:`retsupp.decode.aggregate` -- TSV concat + HP-vs-LP figure.
"""
