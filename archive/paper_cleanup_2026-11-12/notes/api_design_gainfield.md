# Composable gain-field API for braincoder — design sketch

## Motivation

Our current model class tree has gotten unsustainable:

```
DoGAttentionFieldPRF2D
└── DoGDynamicAttentionFieldPRF2D_v3              # +dynamic distractor
    └── _v3_target                                # +target onset
        └── _v3_target_sharedSigma                # +tie σ_dyn = σ_T_dyn
            └── _v3_target_sharedSigma_factorial  # +per-gain sign constraints
                └── _v3_target_sharedSigma_factorial_runPosition  # +per-block-run gains
                └── _v3_target_sharedSigma_oversampled            # +temporal oversampling
```

Each addition is a new subclass, and they don't compose. Adding a
"learning slope" + "factorial signs" + "rectangle paradigm" simultaneously
is a combinatorial explosion of class names.

The underlying model is clean and modular:

```
M(g, t) = 1 + Σ_components C_i(g, t)
```

where each component `C_i` is a simple bilinear product
`gain_i × indicator_i(t) × spatial_i(g)`.

We can express ALL the variants we've built as combinations of a small
set of component types.

## Proposed API

```python
class GainFieldComponent:
    """One additive term in the modulation field M(g, t)."""
    name: str                    # for parameter naming/saving
    locations: ndarray           # (n_loc, 2) — where the spatial bumps sit
    sigma_param: str | float     # name of shared sigma param, or fixed value
    indicator: ndarray | callable # (T, n_loc) — 1 when this component is active per location per TR
    gain_param: str              # name of scalar gain parameter
    sign: str                    # 'free', 'plus', 'minus', 'zero'
    init_value: float            # initial value of the gain
```

Examples:

```python
# Sustained HP/LP (the current setup):
GainFieldComponent(
    name='sus_HP',
    locations=ring_positions,         # 4×2
    sigma_param='sigma_AF',
    indicator=condition_indicator,    # (T, 4) one-hot of HP location per run
    gain_param='g_HP',
    sign='free',
)
GainFieldComponent(
    name='sus_LP',
    locations=ring_positions,
    sigma_param='sigma_AF',
    indicator=1 - condition_indicator,  # one-hot of NON-HP locations per run
    gain_param='g_LP',
    sign='free',
)

# Dynamic distractor terms (split HP / LP):
GainFieldComponent(
    name='dyn_HP',
    locations=ring_positions,
    sigma_param='sigma_dyn',
    indicator=dynamic_indicator * condition_indicator,   # only HP location, only when distractor present
    gain_param='g_HP_dyn',
    sign='free',
)
GainFieldComponent(
    name='dyn_LP',
    locations=ring_positions,
    sigma_param='sigma_dyn',
    indicator=dynamic_indicator * (1 - condition_indicator),
    gain_param='g_LP_dyn',
    sign='free',
)

# Target capture term:
GainFieldComponent(
    name='target',
    locations=ring_positions,
    sigma_param='sigma_dyn',           # ← shared with sigma_dyn (no σ_T_dyn variable)
    indicator=target_indicator,        # (T, 4)
    gain_param='g_T_dyn',
    sign='free',
)
```

Run-position learning is a slight extension: the `gain_param` becomes
a (3,) vector indexed by `run_position(t)`. Either model that as 3
gain components with different indicators, or extend the API:

```python
GainFieldComponent(
    name='sus_HP_per_pos',
    locations=ring_positions,
    sigma_param='sigma_AF',
    indicator=condition_indicator * run_position_one_hot,  # (T, n_loc * 3)
    gain_param='g_HP_pos',                                  # vector of 3
    ...
)
```

## Model class

The fitted PRF model just sums the components:

```python
class GainFieldPRFModel(GaussianPRF2D):  # or DoG, or any base
    def __init__(self, ..., components: list[GainFieldComponent], ...):
        self.components = components
        self._collect_shared_params()  # gather all unique gain/sigma names

    def _modulation_field(self, parameters):
        """Build M(g, t) = 1 + Σ components."""
        M = 1.0
        for c in self.components:
            spatial = self._spatial_gaussian(c.locations, c.sigma)
            temporal = c.indicator
            gain = self._get_gain(c, parameters)
            M += gain * (temporal[:, :, None] * spatial[None, :, :]).sum(axis=1)
        return M

    def _basis_predictions(self, paradigm, parameters):
        # standard pipeline: PRF * paradigm * M, integrate, HRF-convolve
        ...
```

## Model variants become 1-line config changes

| Current model | Component list |
|---|---|
| `_v3` | sus_HP, sus_LP, dyn_HP, dyn_LP |
| `_v3_target` | + target |
| `_v3_target_sharedSigma` | (target uses sigma_dyn instead of sigma_T_dyn) |
| `_v3_target_sharedSigma_factorial` | (gain signs flip per-component via sign='plus'/'minus') |
| `_runPosition` | (sus_HP / sus_LP indicator gets one-hotted by run-position) |
| `_oversampled` | (resampling done in the loader, model unchanged) |

Adding the rectangle paradigm becomes "switch the spatial Gaussian
component for a rotated rectangle" — a different `spatial_kernel`
function attached to the component.

## Migration plan

1. Implement `GainFieldComponent` and `GainFieldPRFModel` as a NEW class
   in `retsupp/modeling/local_models.py` (or `libs/braincoder` if we
   eventually upstream).
2. Verify it reproduces existing fits (numerically) on one subject.
3. Mark existing concrete subclasses as legacy; new fits use composition.
4. Long-term: contribute back to braincoder as a clean API.

## What to do now

This is a moderate refactor (~1-2 days of careful coding) but a
permanent simplification. **Not to be done while we're still iterating
on model variants for VSS.** Worth doing AFTER the talk/paper is in
shape, so we don't break working fits in the middle of analysis.

For the immediate VSS / paper writing phase, keep using the concrete
subclasses; document this design as the next-iteration API.
