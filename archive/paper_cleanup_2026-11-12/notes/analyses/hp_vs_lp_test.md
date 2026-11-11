# Sustained HP-suppression test (g_HP < g_LP)

**Source**: Gaussian static AF (`AttentionFieldPRF2D`,
`derivatives/af_prf_joint_full/`, 240 fits over 30 subjects × 8 ROIs).
Parameters in `notes/af_parameters.tsv`.

## Test design

Per (subject, ROI) we have shared parameters `σ_AF, g_HP, g_LP` from
the joint AF + Gaussian-PRF fit. The HP-specific suppression
hypothesis: HP location is suppressed more than the LP locations.

Two equivalent metrics:
- `g_diff = g_HP − g_LP` (negative ⇒ HP more suppressed)
- `log_ratio = log((1+g_HP) / (1+g_LP))` (scale-invariant; same sign as `g_diff` in practice)

Per-ROI **Wilcoxon paired one-sided** test (alternative='less'),
N = 30 subjects.

## Results (30 subjects, signed-gain mode)

| ROI | n | median `g_diff` | Wilcoxon p (alt = less) | % subj negative |
|---|---|---|---|---|
| **V2** | 30 | **-0.089** | **0.0002** ⭐⭐⭐ | **80%** |
| **V3** | 30 | **-0.134** | **0.0002** ⭐⭐⭐ | **80%** |
| **V3AB** | 30 | -0.115 | **0.05** ⭐ | 63% |
| **VO** | 30 | **-0.242** | **0.003** ⭐⭐ | 80% |
| hV4 | 30 | -0.131 | 0.17 | 57% |
| V1 | 30 | -0.019 | 0.25 | 60% |
| TO | 30 | +0.028 | 0.62 | 47% |
| LO | 30 | +0.043 | 0.85 | 37% |

The same pattern survives at scale-invariant log_ratio (both metrics
correlate at r ≈ 0.95).

## On the σ × |gain| identifiability worry

In V2/V3AB/hV4/TO/VO there is a strong negative Spearman correlation
across subjects within ROI between σ_AF and |g| (e.g. V3AB r_s = -0.78,
p = 0.008). This is a partial identifiability — multiple (σ_AF, gains)
combinations can explain the BOLD similarly. But the rank-based
HP-vs-LP differential per fit is robust to this trade-off: even with
σ-saturated fits, the SIGN of the gain difference holds.

The robustness is also why we get the V3 / V2 result reliably across
30 subjects — 80%-positive-effect is hard to fake from noise alone.

## Reproduction

```bash
~/mambaforge/envs/retsupp/bin/python -m retsupp.visualize.plot_af_parameters \
    --fits-dir /data/ds-retsupp/derivatives/af_prf_joint_full \
    --out notes/af_parameters.pdf \
    --predict-shifts-tsv notes/predict_shifts_cluster.tsv
```

## See also

- [issues/identifiability.md](../issues/identifiability.md) — the σ × |gain| trade-off.
- [analyses/hierarchy_effect.md](hierarchy_effect.md) — the magnitude grows up the visual stream.
- [analyses/sigma_comparison.md](sigma_comparison.md) — sustained vs phasic σ from v3 fits.
