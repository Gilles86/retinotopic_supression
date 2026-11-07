"""One HRF per subject × ROI for the gain-field fits.

The table (``build_roi_hrf_table.py``) holds ``subject, roi, hrf_delay, hrf_dispersion`` from the joint pRF + HRF
fits (``cv/fit_prf_cv_roi_hrf_joint.py``). Every script that builds a gain-field model for the final fits takes the
HRF from here, so fitting, prediction and the fixed-group / leave-one-subject-out code use the same values.
Without a table the canonical HRF is used (delay 4.5 s, dispersion 0.75), as before.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from braincoder.hrf import SPMHRFModel

CANONICAL = {'hrf_delay': 4.5, 'hrf_dispersion': 0.75}


def load_table(path):
    if not path:
        return None
    t = pd.read_csv(Path(path), sep='\t')
    return t.set_index(['subject', 'roi'])[['hrf_delay', 'hrf_dispersion']]


def hrf_values(subject, roi, table=None):
    if table is None:
        return dict(CANONICAL)
    try:
        row = table.loc[(int(subject), roi)]
    except KeyError:
        raise KeyError(f'no HRF for sub-{int(subject):02d} {roi} in the HRF table')
    return {'hrf_delay': float(row.hrf_delay), 'hrf_dispersion': float(row.hrf_dispersion)}


def hrf_model(sub, roi, table=None):
    """The fixed SPM HRF of this subject × ROI (canonical without a table)."""
    v = hrf_values(sub.subject_id, roi, table)
    return SPMHRFModel(tr=sub.get_tr(session=1, run=1), delay=v['hrf_delay'], dispersion=v['hrf_dispersion'])
