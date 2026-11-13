"""Addendum 6 (notes/rt_power_plan_2026-11-13.md): one Bayesian model of single-trial log RT with the trial's LATENT
target, distractor and global gains as predictors, each observed through its BOLD estimate with its sampling
covariance. Note: notes/rt_joint_latent_model_2026-11-13.md.

Model, per correct trial k of participant s (gains: the per-trial pooled estimates of ``collect_trialse``)::

    latent gains   z_k = m_k + diag(τ_s) · u_k,   u_k ~ N(0, R)            (R: 3 × 3 LKJ correlation)
                   m_kj = α_sj + C_k · (γ_j + δ_sj)                         (gain condition effects, group + participant)
    observed       ĝ_k ~ N(z_k, Σ_k)                                         (Σ_k: placebo covariance of the estimate)
    log RT         y_k ~ N(X_k β + a_s + b_s · u_k, σ_s)
                   b_s ~ MVN(b_μ, diag(σ_b) L L' diag(σ_b))                  (correlated participant slopes, LKJ)

u_k is the latent gain's trial-to-trial deviation from its condition mean in units of its within-participant SD, so
b_μ is the group change in log RT per SD of latent gain. The distractor gain exists on distractor trials only (no
u_D term on trials without one). X: distractor condition (HP / LP vs absent), target at HP, target repeat, distractor
repeat, trial number (linear, quadratic), run (1–6) and session — the Bambi formula of the cross-check.

Sampling: the latent gains are integrated out analytically ((ĝ_k, y_k) is jointly Gaussian given the parameters), which
is the same model with ~17k × 3 fewer sampled dimensions; every hierarchical term is non-centred. PyMC, NUTS through
NumPyro (JAX) on CPU.

Subcommands: ``fit`` (``--pool all9`` main; pathway pools for robustness; ``--se-zero`` for the check against Bambi;
``--sim <npz>`` to refit simulated data), ``bambi`` (observed gains as predictors, no measurement error), ``simulate``
(data from the fitted model with known slopes), ``report`` (tables, PPC).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ['T', 'D', 'G']
EXCLUDE = (1, 2)
GAIN_COND = ['dHP', 'dLP', 'tHP', 't_rep', 'd_rep', 'ses2']        # gain condition effects (T, G)
GAIN_COND_D = ['dHP', 'tHP', 't_rep', 'd_rep', 'ses2']             # distractor gain (distractor trials; LP = reference)
FIXED = ['dHP', 'dLP', 'tHP', 't_rep', 'd_rep', 'trial', 'trial2', 'run2', 'run3', 'run4', 'run5', 'run6', 'ses2']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(gains_tsv, behaviour_tsv, pool):
    g = pd.read_csv(gains_tsv, sep='\t')
    g = g[(g['pool'] == pool) & ~g['subject'].isin(EXCLUDE)].copy()
    b = pd.read_csv(behaviour_tsv, sep='\t')
    b = b[~b['subject'].isin(EXCLUDE)]
    key = ['subject', 'session', 'run', 'trial_nr']
    b['trial_nr'] = b['trial_nr'].astype(int)
    d = g.drop(columns=['rt', 'correct']).merge(b[key + ['rt', 'correct', 'dist_cond', 'target_at_hp']], on=key,
                                                how='left', validate='one_to_one')
    assert d['dist_cond'].notna().all(), 'trials without behaviour'
    # Trial types agree between the gain pipeline and the behaviour table.
    tt = d['trial_type'].map({'none': 'Absent', 'HP': 'HP', 'LP': 'LP'})
    assert (tt == d['dist_cond']).all()
    d = d[(d['correct'].astype(str) == 'True') & d['rt'].notna()].copy()
    d['y'] = np.log(d['rt'])
    d['dHP'] = (d['dist_cond'] == 'HP').astype(float)
    d['dLP'] = (d['dist_cond'] == 'LP').astype(float)
    d['tHP'] = d['target_at_hp'].astype(bool).astype(float)
    d['t_rep'] = d['mod_target_repeat'].astype(float)
    d['d_rep'] = d['mod_dist_repeat'].astype(float)
    t = d['trial_nr'].astype(float)
    d['trial'] = (t - t.mean()) / t.std()
    d['trial2'] = d['trial'] ** 2 - (d['trial'] ** 2).mean()
    for r in range(2, 7):
        d[f'run{r}'] = (d['run'] == r).astype(float)
    d['ses2'] = (d['session'] == 2).astype(float)
    d['has_D'] = d['gD'].notna()
    d = d.sort_values(['subject', 'session', 'run', 'trial_nr']).reset_index(drop=True)
    d['sidx'] = pd.factorize(d['subject'], sort=True)[0]
    return d


def gain_arrays(d, se_scale=1.0):
    """Observed gains and their covariance in units of each kind's SD over all included trials."""
    sd = np.array([d['gT'].std(), d['gD'].std(), d['gG'].std()])
    obs = d[['gT', 'gD', 'gG']].to_numpy() / sd
    C = np.zeros((len(d), 3, 3))
    names = {(0, 0): 'vTT', (1, 1): 'vDD', (2, 2): 'vGG', (0, 1): 'cTD', (0, 2): 'cTG', (1, 2): 'cDG'}
    for (i, j), c in names.items():
        v = d[c].to_numpy() / (sd[i] * sd[j])
        C[:, i, j] = v
        C[:, j, i] = v
    C = C * se_scale ** 2
    if se_scale == 0:
        C = C + 1e-4 * np.eye(3)   # SE of 0.01 SD: the check against Bambi
    return np.nan_to_num(obs), np.nan_to_num(C), sd


# ---------------------------------------------------------------------------
# PyMC model (latent gains integrated out)
# ---------------------------------------------------------------------------
def mvn_logp_elementwise(x, mu, cov):
    """Sum over trials of the Gaussian log density; x, mu: lists of n (N,) tensors, cov: n × n nested list of (N,)
    tensors. Cholesky and forward substitution written out per element (fast and batched; n = 3 or 4)."""
    import pytensor.tensor as pt
    n = len(x)
    L = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = cov[i][j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            L[i][j] = pt.sqrt(acc) if i == j else acc / L[j][j]
    w = []
    for i in range(n):
        acc = x[i] - mu[i]
        for k in range(i):
            acc = acc - L[i][k] * w[k]
        w.append(acc / L[i][i])
    return pt.sum(-0.5 * sum(wi ** 2 for wi in w) - sum(pt.log(L[i][i]) for i in range(n)) - 0.5 * n * np.log(2 * np.pi))


def build_model(d, obs, C):
    import pymc as pm
    import pytensor.tensor as pt

    S = d['sidx'].nunique()
    sid = d['sidx'].to_numpy()
    X = d[FIXED].to_numpy()
    Cg = d[GAIN_COND].to_numpy()
    CgD = d[GAIN_COND_D].to_numpy()
    hasD = d['has_D'].to_numpy()
    y = d['y'].to_numpy()
    coords = dict(subject=np.sort(d['subject'].unique()), fixed=FIXED, kind=KINDS, gcond=GAIN_COND, gcondD=GAIN_COND_D)
    with pm.Model(coords=coords) as model:
        # ---- RT part ----
        b0 = pm.Normal('intercept', np.log(0.85), 0.3)
        beta = pm.Normal('beta', 0.0, 0.2, dims='fixed')
        sd_a = pm.HalfNormal('sd_a', 0.3)
        a = pm.Deterministic('a', sd_a * pm.Normal('a_z', 0, 1, dims='subject'), dims='subject')
        mu_ls = pm.Normal('mu_log_sigma', np.log(0.2), 0.5)
        sd_ls = pm.HalfNormal('sd_log_sigma', 0.3)
        sigma = pm.Deterministic('sigma', pt.exp(mu_ls + sd_ls * pm.Normal('sigma_z', 0, 1, dims='subject')),
                                 dims='subject')
        b_mu = pm.Normal('b_mu', 0.0, 0.05, dims='kind')
        chol_b, _, _ = pm.LKJCholeskyCov('chol_b', n=3, eta=2.0, sd_dist=pm.HalfNormal.dist(0.03, shape=3),
                                         compute_corr=True)
        b = pm.Deterministic('b', b_mu + pm.Normal('b_z', 0, 1, dims=('subject', 'kind')) @ chol_b.T,
                             dims=('subject', 'kind'))
        mu_y = b0 + pt.dot(X, beta) + a[sid]

        # ---- latent gains ----
        alpha_mu = pm.Normal('alpha_mu', 0, 1, dims='kind')
        alpha_sd = pm.HalfNormal('alpha_sd', 1, dims='kind')
        alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * pm.Normal('alpha_z', 0, 1, dims=('subject', 'kind')),
                                 dims=('subject', 'kind'))
        gam_T = pm.Normal('gamma_T', 0, 0.5, dims='gcond')
        gam_G = pm.Normal('gamma_G', 0, 0.5, dims='gcond')
        gam_D = pm.Normal('gamma_D', 0, 0.5, dims='gcondD')
        dsd_T = pm.HalfNormal('delta_sd_T', 0.3, dims='gcond')
        dsd_G = pm.HalfNormal('delta_sd_G', 0.3, dims='gcond')
        dsd_D = pm.HalfNormal('delta_sd_D', 0.3, dims='gcondD')
        cT = gam_T + dsd_T * pm.Normal('delta_z_T', 0, 1, dims=('subject', 'gcond'))
        cG = gam_G + dsd_G * pm.Normal('delta_z_G', 0, 1, dims=('subject', 'gcond'))
        cD = gam_D + dsd_D * pm.Normal('delta_z_D', 0, 1, dims=('subject', 'gcondD'))
        mT = alpha[sid, 0] + (Cg * cT[sid]).sum(1)
        mD = alpha[sid, 1] + (CgD * cD[sid]).sum(1)
        mG = alpha[sid, 2] + (Cg * cG[sid]).sum(1)
        mu_ltau = pm.Normal('mu_log_tau', np.log(0.5), 1.0, dims='kind')
        sd_ltau = pm.HalfNormal('sd_log_tau', 0.5, dims='kind')
        tau = pm.Deterministic('tau', pt.exp(mu_ltau + sd_ltau * pm.Normal('tau_z', 0, 1, dims=('subject', 'kind'))),
                               dims=('subject', 'kind'))
        rpk = pm.LKJCorr('R_packed', n=3, eta=2.0)
        R = pt.stack([pt.stack([1.0, rpk[0], rpk[1]]), pt.stack([rpk[0], 1.0, rpk[2]]),
                      pt.stack([rpk[1], rpk[2], 1.0])])
        pm.Deterministic('R', R)

        # ---- joint Gaussian of (ĝ, y) per trial, written out elementwise ----
        Rm = [[1.0, rpk[0], rpk[1]], [rpk[0], 1.0, rpk[2]], [rpk[1], rpk[2], 1.0]]
        tk = [tau[sid, j] for j in range(3)]
        bk = [b[sid, j] for j in range(3)]
        mk = [mT, mD, mG]
        sig2 = sigma[sid] ** 2

        def part(idx, rows):
            """(ĝ_idx, y) on the trials `rows`: Cov(ĝ) = τRτ + Σ, Cov(ĝ, y) = τ R b, Var(y) = b'Rb + σ²."""
            t = [tk[i][rows] for i in idx]
            bb = [bk[i][rows] for i in idx]
            n = len(idx)
            cov = [[None] * (n + 1) for _ in range(n + 1)]
            for a_, i in enumerate(idx):
                for c_, j in enumerate(idx):
                    cov[a_][c_] = t[a_] * Rm[i][j] * t[c_] + C[rows, i, j]
                cov[a_][n] = cov[n][a_] = t[a_] * sum(Rm[i][j] * bb[c_] for c_, j in enumerate(idx))
            cov[n][n] = sum(bb[a_] * Rm[i][j] * bb[c_] for a_, i in enumerate(idx) for c_, j in enumerate(idx)) \
                + sig2[rows]
            x = [obs[rows, i] for i in idx] + [y[rows]]
            mu = [mk[i][rows] for i in idx] + [mu_y[rows]]
            return mvn_logp_elementwise(x, mu, cov)

        p = np.nonzero(hasD)[0]
        q = np.nonzero(~hasD)[0]
        pm.Potential('loglik', part([0, 1, 2], p) + part([0, 2], q))
    return model


def fit(args):
    import arviz as az
    import pymc as pm
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(args.gains, args.behaviour, args.pool)
    if args.sim:
        z = np.load(args.sim)
        d['y'] = z['y']
        obs, C = z['obs'], z['C']
        sd = z['sd']
    else:
        obs, C, sd = gain_arrays(d, 0.0 if args.se_zero else 1.0)
    if args.subjects:
        keep = d['subject'].isin(args.subjects).to_numpy()
        d, obs, C = d[keep].reset_index(drop=True), obs[keep], C[keep]
        d['sidx'] = pd.factorize(d['subject'], sort=True)[0]
    print(f'{len(d)} trials, {d.subject.nunique()} participants, {d.has_D.sum()} with a distractor', flush=True)
    model = build_model(d, obs, C)
    with model:
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
                          nuts_sampler='numpyro', random_seed=args.seed, progressbar=False,
                          nuts_sampler_kwargs=dict(chain_method='parallel'))
    idata.attrs.update(pool=args.pool, se_zero=int(args.se_zero), sim=str(args.sim or ''),
                       gains=str(args.gains), gain_sd=json.dumps(sd.tolist()))
    idata.to_netcdf(out / f'{args.tag}.nc')
    summ = az.summary(idata, var_names=['~a_z', '~sigma_z', '~b_z', '~alpha_z', '~delta_z_T', '~delta_z_G', '~delta_z_D',
                                        '~tau_z', '~a', '~sigma', '~b', '~alpha', '~tau'])
    summ.to_csv(out / f'{args.tag}_summary.tsv', sep='\t')
    div = int(idata.sample_stats['diverging'].sum())
    full = az.summary(idata, var_names=['~R'])
    print(summ.loc[[i for i in summ.index if i.startswith(('b_mu', 'beta', 'intercept', 'mu_log_tau', 'R_packed',
                                                            'chol_b_stds'))]].to_string())
    print(f'divergences {div}, max r_hat {full.r_hat.max():.3f}, min ESS bulk {full.ess_bulk.min():.0f}, '
          f'tail {full.ess_tail.min():.0f}', flush=True)


# ---------------------------------------------------------------------------
# Bambi cross-check: observed gains, no measurement error
# ---------------------------------------------------------------------------
BAMBI_FORMULA = ('log_rt ~ dist_cond + target_cond + target_repeat + dist_repeat + trial + trial2 + C(run) + C(session) '
                 '+ g_target + g_dist + g_global + (1 + g_target + g_dist + g_global | subject)')


def observed_predictors(d):
    """Observed gains as Bambi predictors: per participant, residual of each gain on its condition design (as the
    latent model's condition means), divided by its SD; distractor gain 0 on trials without a distractor."""
    out = {}
    for k, col, cols in (('g_target', 'gT', GAIN_COND), ('g_dist', 'gD', GAIN_COND_D), ('g_global', 'gG', GAIN_COND)):
        v = np.zeros(len(d))
        for s, ix in d.groupby('subject').groups.items():
            dd = d.loc[ix]
            ok = dd[col].notna().to_numpy()
            Xc = dd.loc[ok, cols].to_numpy()
            Xs = np.column_stack([np.ones(ok.sum()), Xc[:, Xc.std(0) > 0]])
            yy = dd.loc[ok, col].to_numpy()
            r = yy - Xs @ np.linalg.lstsq(Xs, yy, rcond=None)[0]
            v[np.asarray(ix)[ok]] = r / r.std()
        out[k] = v
    return pd.DataFrame(out, index=d.index)


def bambi(args):
    import arviz as az
    import bambi as bmb
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(args.gains, args.behaviour, args.pool)
    df = pd.concat([d[['subject', 'run', 'session', 'trial', 'trial2']], observed_predictors(d)], axis=1)
    df['log_rt'] = d['y']
    df['dist_cond'] = pd.Categorical(d['dist_cond'], categories=['Absent', 'HP', 'LP'])
    df['target_cond'] = pd.Categorical(np.where(d['tHP'] > 0, 'HP', 'LP'), categories=['LP', 'HP'])
    df['target_repeat'] = d['t_rep']
    df['dist_repeat'] = d['d_rep']
    df['subject'] = df['subject'].astype(str)
    m = bmb.Model(BAMBI_FORMULA, df)
    print(m, flush=True)
    idata = m.fit(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
                  inference_method='numpyro', random_seed=args.seed, progressbar=False)
    idata.to_netcdf(out / f'{args.tag}.nc')
    summ = az.summary(idata, var_names=['~_offset'], filter_vars='like')
    summ.to_csv(out / f'{args.tag}_summary.tsv', sep='\t')
    print(summ.loc[[i for i in summ.index if '|' not in i]].to_string())
    print(f"divergences {int(idata.sample_stats['diverging'].sum())}, max r_hat {summ.r_hat.max():.3f}, "
          f"min ESS bulk {summ.ess_bulk.min():.0f}")


# ---------------------------------------------------------------------------
# Simulation from the fitted model (parameter recovery) and posterior predictive draws
# ---------------------------------------------------------------------------
def param_means(idata):
    p = idata.posterior
    return {k: p[k].mean(('chain', 'draw')).values for k in p.data_vars}


def predictive_pieces(d, P):
    """Condition means of the latent gains and of log RT, for one parameter set P (dict of arrays)."""
    sid = d['sidx'].to_numpy()
    X = d[FIXED].to_numpy()
    mu_y = P['intercept'] + X @ P['beta'] + P['a'][sid]
    cT = P['gamma_T'] + P['delta_sd_T'] * P['delta_z_T']
    cG = P['gamma_G'] + P['delta_sd_G'] * P['delta_z_G']
    cD = P['gamma_D'] + P['delta_sd_D'] * P['delta_z_D']
    m = np.column_stack([P['alpha'][sid, 0] + (d[GAIN_COND].to_numpy() * cT[sid]).sum(1),
                         P['alpha'][sid, 1] + (d[GAIN_COND_D].to_numpy() * cD[sid]).sum(1),
                         P['alpha'][sid, 2] + (d[GAIN_COND].to_numpy() * cG[sid]).sum(1)])
    return mu_y, m


def simulate_once(d, P, C, rng):
    """(obs gains, y) from the generative model with parameters P; the measurement noise uses each trial's Σ_k."""
    sid = d['sidx'].to_numpy()
    hasD = d['has_D'].to_numpy()
    mu_y, m = predictive_pieces(d, P)
    L = np.linalg.cholesky(P['R'])
    u = rng.standard_normal((len(d), 3)) @ L.T
    z = m + P['tau'][sid] * u
    e = np.einsum('nij,nj->ni', np.linalg.cholesky(C + 1e-12 * np.eye(3)), rng.standard_normal((len(d), 3)))
    obs = z + e
    bu = P['b'][sid] * u
    bu[~hasD, 1] = 0.0
    y = mu_y + bu.sum(1) + P['sigma'][sid] * rng.standard_normal(len(d))
    obs[~hasD, 1] = 0.0
    return obs, y


def simulate(args):
    import arviz as az
    d = load_data(args.gains, args.behaviour, args.pool)
    obs, C, sd = gain_arrays(d)
    idata = az.from_netcdf(args.trace)
    P = param_means(idata)
    true_b_mu = np.array(args.true_b_mu)
    # Participant slopes: the fitted participant deviations around the new group mean.
    P['b'] = P['b'] - P['b_mu'][None] + true_b_mu[None]
    P['b_mu'] = true_b_mu
    obs_s, y_s = simulate_once(d, P, C, np.random.default_rng(args.seed))
    np.savez(args.out, obs=obs_s, y=y_s, C=C, sd=sd, true_b_mu=true_b_mu, true_b=P['b'],
             true_beta=P['beta'], true_tau=P['tau'], true_R=P['R'], true_gamma_T=P['gamma_T'],
             true_gamma_D=P['gamma_D'], true_gamma_G=P['gamma_G'])
    print(f'Saved {args.out}: b_mu = {true_b_mu}')


# ---------------------------------------------------------------------------
# Report: slopes, condition effects, PPC, recovery, Bambi comparison
# ---------------------------------------------------------------------------
def hdi_row(x, rt_bar, name, unit='log'):
    import arviz as az
    x = np.asarray(x).ravel()
    lo, hi = az.hdi(x, hdi_prob=0.95)
    r = dict(term=name, mean_log=x.mean(), hdi_lo_log=lo, hdi_hi_log=hi,
             mean_pct=100 * np.expm1(x).mean(), hdi_lo_pct=100 * np.expm1(lo), hdi_hi_pct=100 * np.expm1(hi),
             mean_ms=rt_bar * np.expm1(x).mean() * 1000, hdi_lo_ms=rt_bar * np.expm1(lo) * 1000,
             hdi_hi_ms=rt_bar * np.expm1(hi) * 1000, p_neg=(x < 0).mean(), p_pos=(x > 0).mean())
    return r


def draw_params(post, c, i):
    return {k: post[k].values[c, i] for k in post.data_vars}


def report(args):
    import arviz as az
    out = Path(args.out_dir)
    d = load_data(args.gains, args.behaviour, args.pool)
    rt_bar = d.groupby('subject')['rt'].mean().mean()
    rows = []
    for tag in args.traces:
        idata = az.from_netcdf(out / f'{tag}.nc')
        post = idata.posterior
        stack = lambda v: post[v].stack(s=('chain', 'draw')).values
        b = stack('b_mu')
        for j, k in enumerate(KINDS):
            rows.append(dict(fit=tag, **hdi_row(b[j], rt_bar, f'slope_{k}')))
        tau_s = stack('tau')                                # (S, 3, draws)
        b_s = stack('b')
        for j, k in enumerate(KINDS):
            # log RT per unit of latent gain, in units of the observed gain's SD over all trials (b_sj / τ_sj,
            # averaged over participants): the quantity the data pin down when τ is weakly identified
            rows.append(dict(fit=tag, **hdi_row((b_s[:, j] / tau_s[:, j]).mean(0), rt_bar, f'slope_per_obsSD_{k}')))
        beta = stack('beta')
        fx = list(post['beta'].coords['fixed'].values)
        B = {n: beta[fx.index(n)] for n in fx}
        for name, x in [('HP vs LP distractor', B['dHP'] - B['dLP']), ('LP distractor vs absent', B['dLP']),
                        ('HP distractor vs absent', B['dHP']), ('target at HP vs not', B['tHP']),
                        ('target repeat', B['t_rep']), ('distractor repeat', B['d_rep'])]:
            rows.append(dict(fit=tag, **hdi_row(x, rt_bar, name)))
        s = az.summary(idata, var_names=['~R'])
        rows.append(dict(fit=tag, term='convergence', max_rhat=s.r_hat.max(), min_ess_bulk=s.ess_bulk.min(),
                         min_ess_tail=s.ess_tail.min(), divergences=int(idata.sample_stats['diverging'].sum()),
                         n_draws=b.shape[1]))
        # Latent gain scale, reliability and correlation
        tau = stack('tau')                                  # (S, 3, draws)
        obs, C, sd = gain_arrays(d)
        se2 = np.array([np.median(C[d['has_D'].to_numpy() | (j != 1), j, j]) for j in range(3)])
        for j, k in enumerate(KINDS):
            t2 = (tau[:, j] ** 2).mean(0)
            rows.append(dict(fit=tag, term=f'reliability_{k}', mean_log=float(np.mean(t2 / (t2 + se2[j])))))
        Rm = post['R'].mean(('chain', 'draw')).values
        rows.append(dict(fit=tag, term='latent_R', note=json.dumps(np.round(Rm, 3).tolist())))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f'{args.tag}_results.tsv', sep='\t', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(tab.round(4).to_string())

    if args.ppc:
        idata = az.from_netcdf(out / f'{args.traces[0]}.nc')
        post = idata.posterior
        obs, C, sd = gain_arrays(d)
        rng = np.random.default_rng(3)
        nc, nd = post.sizes['chain'], post.sizes['draw']
        picks = rng.choice(nc * nd, size=args.n_ppc, replace=False)
        cond = d['dist_cond'] + '|target ' + np.where(d['tHP'] > 0, 'HP', 'LP')
        stats, subj, fig1 = [], [], []
        def summarise(y, g, who):
            rt = np.exp(y) * 1000
            for c_, ix in d.groupby(cond).groups.items():
                ix = np.asarray(ix)
                r = dict(source=who, cond=c_, mean_rt=rt[ix].mean(), q10=np.quantile(rt[ix], .1),
                         q50=np.quantile(rt[ix], .5), q90=np.quantile(rt[ix], .9), sd_log_rt=y[ix].std())
                for j, k in enumerate(KINDS):
                    if k == 'D' and c_.startswith('Absent'):
                        continue
                    r[f'mean_g{k}'] = g[ix, j].mean()
                    r[f'sd_g{k}'] = g[ix, j].std()
                stats.append(r)
            m = pd.Series(rt).groupby([d['subject'], d['dist_cond']]).mean()
            subj.append(m.rename(who))
            # Fig. 1 contrasts: per-participant condition means, difference, mean over participants (ms)
            mm = m.unstack()
            onD = d['dist_cond'] != 'Absent'
            tt = pd.Series(rt[onD.to_numpy()]).groupby([d.loc[onD, 'subject'].to_numpy(),
                                                        d.loc[onD, 'tHP'].to_numpy()]).mean().unstack()
            rep = pd.Series(rt).groupby([d['subject'], d['t_rep']]).mean().unstack()
            drep = pd.Series(rt[onD.to_numpy()]).groupby([d.loc[onD, 'subject'].to_numpy(),
                                                          d.loc[onD, 'd_rep'].to_numpy()]).mean().unstack()
            fig1.append(dict(source=who, hp_vs_lp=(mm['HP'] - mm['LP']).mean(), lp_vs_absent=(mm['LP'] - mm['Absent']).mean(),
                             hp_vs_absent=(mm['HP'] - mm['Absent']).mean(), target_hp_vs_lp=(tt[1.0] - tt[0.0]).mean(),
                             target_repeat=(rep[1.0] - rep[0.0]).mean(), dist_repeat=(drep[1.0] - drep[0.0]).mean()))
        summarise(d['y'].to_numpy(), obs, 'observed')
        for n, pk in enumerate(picks):
            P = draw_params(post, pk // nd, pk % nd)
            o_s, y_s = simulate_once(d, P, C, rng)
            summarise(y_s, o_s, f'sim{n}')
        st = pd.DataFrame(stats)
        ob = st[st.source == 'observed'].set_index('cond')
        sim = st[st.source != 'observed']
        cols = [c for c in st.columns if c not in ('source', 'cond')]
        lo = sim.groupby('cond')[cols].quantile(.025)
        hi = sim.groupby('cond')[cols].quantile(.975)
        mid = sim.groupby('cond')[cols].mean()
        ppc = pd.concat({'observed': ob[cols], 'pred_mean': mid, 'pred_lo': lo, 'pred_hi': hi}, axis=1)
        ppc.to_csv(out / f'{args.tag}_ppc_conditions.tsv', sep='\t')
        inside = ((ob[cols] >= lo) & (ob[cols] <= hi))
        F = pd.DataFrame(fig1).set_index('source')
        fo, fs = F.loc['observed'], F.drop(index='observed')
        f1 = pd.DataFrame(dict(observed=fo, pred_mean=fs.mean(), pred_lo=fs.quantile(.025), pred_hi=fs.quantile(.975)))
        f1.to_csv(out / f'{args.tag}_ppc_fig1_contrasts.tsv', sep='\t')
        print('Fig. 1 contrasts (ms, mean over participants of within-participant differences):')
        print(f1.round(1).to_string())
        S = pd.concat(subj, axis=1)
        sl, sh = S.drop(columns='observed').quantile(.025, axis=1), S.drop(columns='observed').quantile(.975, axis=1)
        cov_subj = float(((S['observed'] >= sl) & (S['observed'] <= sh)).mean())
        print(ppc.round(3).T.to_string())
        print('inside 95% band per statistic (over the 6 conditions):')
        print(inside.mean().round(2).to_string())
        print(f'participant x distractor-condition mean RT coverage: {cov_subj:.3f} ({len(S)} cells)')
        json.dump(dict(coverage_participant_condition=cov_subj, inside=inside.mean().to_dict(), n_ppc=args.n_ppc),
                  open(out / f'{args.tag}_ppc_coverage.json', 'w'), indent=1)


def recovery(args):
    import arviz as az
    out = Path(args.out_dir)
    rows = []
    for sim, tag in zip(args.sims, args.traces):
        z = np.load(sim)
        post = az.from_netcdf(out / f'{tag}.nc').posterior
        def add(name, true, draws):
            draws = np.asarray(draws)
            lo, hi = az.hdi(draws.ravel(), hdi_prob=0.95)
            rows.append(dict(fit=tag, parameter=name, true=float(true), mean=float(draws.mean()), hdi_lo=lo, hdi_hi=hi,
                             covered=bool(lo <= true <= hi)))
        for j, k in enumerate(KINDS):
            add(f'b_mu[{k}]', z['true_b_mu'][j], post['b_mu'].values[..., j])
            add(f'mean tau[{k}]', z['true_tau'][:, j].mean(), post['tau'].values[..., j].mean(-1))
        fx = list(post['beta'].coords['fixed'].values)
        for i, n in enumerate(fx):
            add(f'beta[{n}]', z['true_beta'][i], post['beta'].values[..., i])
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            add(f'R[{KINDS[i]},{KINDS[j]}]', z['true_R'][i, j], post['R'].values[..., i, j])
        tb = z['true_b']
        pb = post['b'].mean(('chain', 'draw')).values
        for j, k in enumerate(KINDS):
            rows.append(dict(fit=tag, parameter=f'participant b[{k}] r(true, est)',
                             mean=float(np.corrcoef(tb[:, j], pb[:, j])[0, 1])))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f'{args.tag}_recovery.tsv', sep='\t', index=False)
    print(tab.round(4).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['fit', 'bambi', 'simulate', 'report', 'recovery'])
    p.add_argument('--gains', required=True)
    p.add_argument('--behaviour', default='notes/data/behaviour_trials_2026-11-11.tsv.gz')
    p.add_argument('--pool', default='all9')
    p.add_argument('--out-dir', default='.')
    p.add_argument('--tag', default='main')
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=1000)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--target-accept', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--se-zero', action='store_true')
    p.add_argument('--sim', default=None, help='npz from `simulate`: refit simulated gains and RTs')
    p.add_argument('--subjects', type=int, nargs='*', default=None, help='smoke tests: only these participants')
    p.add_argument('--trace', default=None)
    p.add_argument('--true-b-mu', type=float, nargs=3, default=[0.01, -0.02, 0.0])
    p.add_argument('--out', default=None)
    p.add_argument('--traces', nargs='*', default=[])
    p.add_argument('--sims', nargs='*', default=[])
    p.add_argument('--ppc', action='store_true')
    p.add_argument('--n-ppc', type=int, default=200)
    a = p.parse_args()
    {'fit': fit, 'bambi': bambi, 'simulate': simulate, 'report': report, 'recovery': recovery}[a.cmd](a)


if __name__ == '__main__':
    os.environ.setdefault('XLA_FLAGS', f'--xla_force_host_platform_device_count={os.environ.get("SLURM_CPUS_PER_TASK", 4)}')
    main()
