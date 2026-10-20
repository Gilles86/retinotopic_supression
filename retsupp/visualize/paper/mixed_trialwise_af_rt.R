# Mixed-effects version of the trial-by-trial AF gain -> RT analysis.
#
# Input: notes/data/trialwise_af_trial_table.tsv.gz (from
#   analyze_trialwise_af_gains.py collect): one row per (subject, ROI, valid
#   trial), RT and gains z-scored within (session, run, trial type).
#
# Two models per ROI (and per ROI group, gains averaged over the group's ROIs
# per trial and re-z-scored):
#   specific: logrt ~ T + G + D + Ddiff + priming covariates
#             D     = distractor gain, HP and LP trials combined (mean slope)
#             Ddiff = HP - LP difference in the distractor slope
#   overall:  logrt ~ A + priming covariates
#             A     = z(mean of the trial's z-scored gains) -- overall trial
#                     response amplitude, identifiable even where T/G/D are not
# Random effects: per-subject intercept + correlated slopes for every gain
# (lme4); if that fit is singular, refit with uncorrelated slopes (||).
# p-values from lmerTest (Satterthwaite df), BH-FDR across ROIs (or across
# the three groups) per term.
#
# Also writes the median within-subject correlation between per-trial
# gains per ROI: where it is ~1 (IPS/SPL1/FEF) T, G and D can't be told
# apart and only the overall model is interpretable.
#
# Usage (needs lme4 + lmerTest; installed in the rbayes env):
#   ~/mambaforge/envs/rbayes/bin/Rscript retsupp/visualize/paper/mixed_trialwise_af_rt.R \
#       notes/data/trialwise_af

suppressMessages({library(lme4); library(lmerTest)})
args <- commandArgs(trailingOnly = TRUE)
prefix <- if (length(args)) args[1] else 'notes/data/trialwise_af'
tt <- read.delim(gzfile(paste0(prefix, '_trial_table.tsv.gz')))

roi_group <- c(V1 = 'early', V2 = 'early', V3 = 'early',
               hV4 = 'ventral/lateral', VO = 'ventral/lateral',
               LO = 'ventral/lateral', TO = 'ventral/lateral',
               V3AB = 'dorsal', IPS = 'dorsal', SPL1 = 'dorsal', FEF = 'dorsal')
cell <- c('subject', 'session', 'run', 'trial_type')
covs <- 'dist_repeat + target_repeat + target_at_prev_dist'
zc <- function(x) { s <- sd(x); if (is.na(s) || s == 0) x * 0 else (x - mean(x)) / s }
zcell <- function(x, d) ave(x, d[cell], FUN = zc)

prep <- function(d) {
  dist <- d$trial_type != 'none'
  d$Ddiff <- d$D * ifelse(d$trial_type == 'HP', 0.5, ifelse(dist, -0.5, 0))
  n_g <- 2 + dist
  d$A <- zcell((d$T + d$G + d$D) / n_g, d)
  d
}

fit <- function(d, unit, kind) {
  terms <- if (kind == 'specific') c('T', 'G', 'D', 'Ddiff') else 'A'
  slopes <- if (kind == 'specific') 'T + G + D' else 'A'
  fixed <- paste('logrt ~', paste(terms, collapse = ' + '), '+', covs)
  ctrl <- lmerControl(optimizer = 'bobyqa', calc.derivs = FALSE)
  re <- 'correlated'
  m <- lmer(as.formula(paste0(fixed, ' + (1 + ', slopes, ' | subject)')),
            data = d, REML = TRUE, control = ctrl)
  if (isSingular(m)) {
    re <- 'uncorrelated'
    m <- lmer(as.formula(paste0(fixed, ' + (1 + ', slopes, ' || subject)')),
              data = d, REML = TRUE, control = ctrl)
  }
  tab <- coef(summary(m))[terms, , drop = FALSE]
  data.frame(unit = unit, model = kind, term = terms, beta = tab[, 'Estimate'],
             se = tab[, 'Std. Error'], df = tab[, 'df'], t = tab[, 't value'],
             p = tab[, 'Pr(>|t|)'], n = length(unique(d$subject)),
             random = re, singular = isSingular(m), row.names = NULL)
}

tt <- prep(tt)
rows <- list()
for (roi in names(roi_group)) {
  d <- tt[tt$roi == roi, ]
  for (kind in c('specific', 'overall'))
    rows[[length(rows) + 1]] <- cbind(level = 'roi', fit(d, roi, kind))
}
# ROI groups: average each trial's gains over the group's ROIs, re-z-score.
tt$group <- roi_group[tt$roi]
key <- c('subject', 'trial_idx', 'session', 'run', 'trial_type', 'group')
g <- aggregate(tt[c('T', 'G', 'D', 'logrt', 'dist_repeat', 'target_repeat',
                    'target_at_prev_dist')], tt[key], mean)
for (v in c('T', 'G')) g[[v]] <- zcell(g[[v]], g)
g$D <- ifelse(g$trial_type == 'none', 0, zcell(g$D, g))
g <- prep(g)
for (grp in unique(roi_group)) {
  d <- g[g$group == grp, ]
  for (kind in c('specific', 'overall'))
    rows[[length(rows) + 1]] <- cbind(level = 'group', fit(d, grp, kind))
}
out <- do.call(rbind, rows)
out$q <- ave(out$p, out$level, out$model, out$term,
             FUN = function(p) p.adjust(p, 'BH'))
write.table(out, paste0(prefix, '_mixed_stats.tsv'), sep = '\t',
            quote = FALSE, row.names = FALSE)

# Identifiability: median within-subject r between per-trial gains.
r <- do.call(rbind, lapply(split(tt, list(tt$roi, tt$subject), drop = TRUE),
  function(x) {
    xd <- x[x$trial_type != 'none', ]
    data.frame(roi = x$roi[1], subject = x$subject[1],
               r_TG = cor(x$T, x$G), r_TD = cor(xd$T, xd$D),
               r_GD = cor(xd$G, xd$D))
  }))
ident <- aggregate(r[c('r_TG', 'r_TD', 'r_GD')], r['roi'], median)
write.table(ident, paste0(prefix, '_gain_collinearity.tsv'), sep = '\t',
            quote = FALSE, row.names = FALSE)

options(width = 140)
print(format(out[, c('level', 'unit', 'model', 'term', 'beta', 'df', 't', 'p', 'q', 'random', 'singular')],
             digits = 3), row.names = FALSE)
print(ident, digits = 2)
