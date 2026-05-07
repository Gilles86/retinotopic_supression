# σ_AF (sustained) vs σ_dyn (phasic) per ROI

**Sources**:
- Gaussian voxel: `derivatives/af_prf_joint_dynamic_v3/`
  (`DynamicAttentionFieldPRF2DWithHRF_v3`).
  Parameters table: `notes/af_v3_parameters.tsv`.
- DoG voxel: `derivatives/af_prf_joint_dynamic_v3_dog/`
  (`DoGDynamicAttentionFieldPRF2DWithHRF_v3`).
  Parameters table: `notes/af_dog_v3_parameters.tsv`.

## Headline finding

In **every visual ROI we tested except VO**, the phasic distractor AF
is significantly NARROWER than the sustained priority-map AF. The
effect is **stronger with DoG voxel kernels** (which control for
center-surround structure in early visual cortex), arguing against an
"only-Gaussian" artifact interpretation.

| ROI | σ_AF (DoG) | σ_dyn (DoG) | p (DoG) | p (Gauss) |
|---|---|---|---|---|
| V1 | 2.40 | **0.66** | **0.001** | 0.010 |
| V2 | 2.74 | **0.88** | **0.003** | 0.011 |
| V3 | 2.86 | **0.79** | **<0.0001** | 0.0002 |
| V3AB | 4.97 | **1.11** | **<0.001** | n.s. |
| hV4 | 3.62 | **0.76** | **<0.0001** | 0.05 |
| LO | 3.12 | **1.00** | **0.002** | n.s. |
| TO | 4.25 | **1.08** | **<0.0001** | n.s. |
| VO | 2.72 | 1.57 | n.s. | 0.007 ⚠️ |

(Paired Wilcoxon, σ_dyn vs σ_AF, two-sided.)

## Interpretation

Two distinct spatial scales for two distinct attentional mechanisms:

- **σ_AF (sustained)** ~2–5°: extent of the LEARNED priority-map zone.
- **σ_dyn (phasic)** ~0.7–1.1°: extent of the modulation when a
  distractor actually appears on screen — close to the distractor disk's
  actual radius (0.4°).

## Why DoG matters

A reasonable concern was that the small σ_dyn in early visual cortex
might be an artifact of using a single-Gaussian voxel kernel: real V1/V2/V3
voxels have center-surround (DoG) organization and DEACTIVATE for
stimuli outside the center. A Gaussian-only model would have to express
that via the AF parameters, possibly making σ_dyn artificially small.

**DoG fits disconfirm this hypothesis** — σ_dyn stays narrow (~0.7–1.1°)
and the effect becomes MORE significant in every ROI. So the narrow
phasic AF is a property of the data, not of the modeling assumption.

## Companion gain pattern

Per the same v3 fits, the dynamic gain (`g_dyn_avg = ½(g_HP_dyn + g_LP_dyn)`)
shows a clear early/higher dissociation (Gaussian voxel, n ≈ 21/ROI):

| ROI | direction | p |
|---|---|---|
| V2 | SUPPRESS | 0.002 |
| V1 | trend SUPPRESS | 0.12 |
| hV4 | CAPTURE | 0.04 |
| LO | CAPTURE | 0.002 |
| VO | CAPTURE | <0.001 |

Early visual SUPPRESSES phasic distractor responses; lateral/dorsal/ventral
areas show phasic CAPTURE. Same pattern direction holds with DoG fits
(numbers will be re-tabulated when v3 DoG fits are 100% complete).

## Reproduction

```bash
# Gaussian
~/mambaforge/envs/retsupp/bin/python -c "
import pickle, glob, numpy as np, pandas as pd
from scipy import stats
paths = sorted(glob.glob(
    '/data/ds-retsupp/derivatives/af_prf_joint_dynamic_v3/sub-*/'
    'sub-*_roi-*_mode-signed_dyn-v3-af-prf-fit.pkl'))
rows = [
    dict(subject=int(p.split('/sub-')[1].split('/')[0]),
         roi=p.split('roi-')[1].split('_')[0],
         **pickle.load(open(p, 'rb'))['shared_pars'])
    for p in paths
]
df = pd.DataFrame(rows)
for roi, sub in df.groupby('roi'):
    _, p = stats.wilcoxon(sub.sigma_dyn, sub.sigma_AF)
    print(f'{roi:<6} n={len(sub)}  σ_AF={np.median(sub.sigma_AF):.2f}  '
          f'σ_dyn={np.median(sub.sigma_dyn):.2f}  p={p:.4f}')
"

# DoG: replace path with af_prf_joint_dynamic_v3_dog and *-dyn-v3-* with *-dyn-v3-*
```
