#!/usr/bin/env python3
'''P10: ten-fold same-fold paired analysis for the component-ablation grid.

Reads results/repro_<model>_fold<f>/test_metrics.json (fold 10 for the three
variants that predate the repro_* grid comes from the ablation_* directories).

Protocol is pre-registered in records/01_实验思路.md, entry 013:
  primary family   = 4 comparisons among the four models measured at n=10
  secondary family = 3 comparisons involving ab_improved (still n=3)
  paired unit      = fold; delta computed inside the SAME test fold
  Holm-Bonferroni  = applied within each family separately

Usage:
  python3 experiments/31_p10_report.py                 # full report
  python3 experiments/31_p10_report.py --allow-partial # progress check
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = Path('/mnt/ECG-SNN-LowPower/results')
OUT_JSON = ROOT / 'p10_summary.json'

MODELS = ['ab_full', 'ab_shallow', 'ab_polarity', 'ab_shallow_pol', 'ab_improved']
FOLDS = list(range(10, 0, -1))

FOLD10_DIR = {
    'ab_full': 'ablation_full',
    'ab_shallow': 'ablation_shallow_3blocks',
    'ab_polarity': 'ablation_encoder_polarity',
    'ab_shallow_pol': 'repro_ab_shallow_pol_fold10',
    'ab_improved': 'repro_ab_improved_fold10',
}

PRIMARY = [
    ('ab_shallow', 'ab_full'),
    ('ab_polarity', 'ab_full'),
    ('ab_shallow_pol', 'ab_full'),
    ('ab_shallow_pol', 'ab_shallow'),
]
SECONDARY = [
    ('ab_improved', 'ab_full'),
    ('ab_improved', 'ab_shallow'),
    ('ab_improved', 'ab_shallow_pol'),
]

PARAMS = {
    'ab_full': 1893768,
    'ab_shallow': 528392,
    'ab_polarity': 1893768,
    'ab_shallow_pol': 528392,
    'ab_improved': 528133,
}


def run_dir(model, fold):
    if fold == 10:
        return ROOT / FOLD10_DIR[model]
    return ROOT / ('repro_%s_fold%d' % (model, fold))


def load(model, fold):
    p = run_dir(model, fold) / 'test_metrics.json'
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return {
        'auroc': float(m['macro_auroc']),
        'auprc': float(m['macro_auprc']),
        'per_label': {k: float(v['auroc']) for k, v in m.get('per_label', {}).items()},
    }


def mean_std(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def paired_stats(a, b, folds):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else float('nan')
    df = n - 1
    tcrit = float(stats.t.ppf(0.975, df)) if df > 0 else float('nan')
    if n > 1:
        t_val, p_val = stats.ttest_rel(a, b)
    else:
        t_val, p_val = float('nan'), float('nan')
    dz = (m / sd) if sd > 0 else None
    out = {
        'folds': folds,
        'per_fold_delta': [round(float(x), 6) for x in d],
        'delta_mean': round(m, 6),
        'delta_std': round(sd, 6),
        'delta_sem': round(float(sem), 6),
        'ci95_low': round(m - tcrit * sem, 6),
        'ci95_high': round(m + tcrit * sem, 6),
        't': round(float(t_val), 4),
        'df': df,
        'p': round(float(p_val), 4),
        'cohens_dz': round(float(dz), 3) if dz is not None else None,
        'n_folds_won': int((d > 0).sum()),
        'n_folds': n,
        'significant_at_0.05': bool(p_val < 0.05),
    }
    if n > 1 and sd > 0:
        out['mdd_80pc'] = round(mdd(n, sd, 0.05, 0.80), 6)
        out['mdd_90pc'] = round(mdd(n, sd, 0.05, 0.90), 6)
    return out


def mdd(n, sd, alpha, power):
    '''Exact minimum detectable difference under the noncentral t.'''
    df = n - 1
    tcrit = float(stats.t.ppf(1 - alpha / 2, df))
    se = sd / np.sqrt(n)

    def f(delta):
        nc = delta / se
        return float(stats.nct.sf(tcrit, df, nc)) - power

    return float(brentq(f, 1e-9, 100 * sd))


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def fmt_p(p):
    return '<0.001' if p < 0.001 else ('%.3f' % p)


def build_family(name, comps, table, alpha=0.05):
    results = {}
    pvals = []
    keys = []
    for var, base in comps:
        shared = [f for f in FOLDS
                  if table.get(var, {}).get(f) and table.get(base, {}).get(f)]
        if len(shared) < 2:
            continue
        a = [table[var][f]['auroc'] for f in shared]
        b = [table[base][f]['auroc'] for f in shared]
        r = paired_stats(a, b, shared)
        r['variant'] = var
        r['baseline'] = base
        results['%s - %s' % (var, base)] = r
        pvals.append(r['p'])
        keys.append('%s - %s' % (var, base))
    if pvals:
        adj = holm(pvals)
        for k, pa in zip(keys, adj):
            results[k]['p_holm'] = round(float(pa), 4)
            results[k]['significant_holm'] = bool(pa < alpha)
    return {'comparisons': results, 'family_size': len(pvals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--out', default=str(OUT_JSON))
    args = ap.parse_args()

    table = {}
    for m in MODELS:
        table[m] = {}
        for f in FOLDS:
            r = load(m, f)
            if r is not None:
                table[m][f] = r

    complete = all(len(table[m]) == 10 for m in MODELS[:4])
    if not complete and not args.allow_partial:
        print('INCOMPLETE: primary models have %s folds; use --allow-partial'
              % {m: len(table[m]) for m in MODELS[:4]}, file=sys.stderr)
        return 2

    summary = {
        'protocol': 'same-fold paired delta, paired unit = test fold, '
                    'Holm-Bonferroni within family; pre-registered in '
                    'records/01_实验思路.md entry 013',
        'fold_table': {},
        'families': {},
    }

    print('=' * 78)
    print('P10 ten-fold same-fold paired analysis')
    print('=' * 78)

    for m in MODELS:
        folds = sorted(table[m].keys(), reverse=True)
        aurocs = [table[m][f]['auroc'] for f in folds]
        auprcs = [table[m][f]['auprc'] for f in folds]
        mu, sd = mean_std(aurocs)
        mu_p, sd_p = mean_std(auprcs)
        summary['fold_table'][m] = {
            'folds': folds,
            'per_fold_auroc': {('fold%d' % f): round(table[m][f]['auroc'], 6) for f in folds},
            'per_fold_auprc': {('fold%d' % f): round(table[m][f]['auprc'], 6) for f in folds},
            'auroc_mean': round(mu, 6),
            'auroc_std': round(sd, 6),
            'auprc_mean': round(mu_p, 6),
            'auprc_std': round(sd_p, 6),
            'params': PARAMS[m],
        }
        print('% -18s n=%2d  AUROC %.4f +- %.4f  AUPRC %.4f +- %.4f'
              % (m, len(folds), mu, sd, mu_p, sd_p))
        print('                   folds: %s' % ' '.join(str(f) for f in folds))

    print('')
    for fam_name, comps in [('primary', PRIMARY), ('secondary', SECONDARY)]:
        fam = build_family(fam_name, comps, table)
        summary['families'][fam_name] = fam
        print('-' * 78)
        print('%s family (m = %d, Holm within family)' % (fam_name, fam['family_size']))
        print('-' * 78)
        print('%-34s %8s %20s %6s %7s %7s %5s'
              % ('comparison', 'delta', '95% CI', 'dz', 't', 'p', 'Holm'))
        for k, r in fam['comparisons'].items():
            ci = '[%+.2f, %+.2f]' % (r['ci95_low'] * 100, r['ci95_high'] * 100)
            print('%-34s %+7.2fpp %20s %6s %7.2f %7s %5s  won %d/%d'
                  % (k, r['delta_mean'] * 100, ci,
                     ('%.2f' % r['cohens_dz']) if r['cohens_dz'] is not None else 'na',
                     r['t'], fmt_p(r['p']), fmt_p(r['p_holm']),
                     r['n_folds_won'], r['n_folds']))
            if 'mdd_80pc' in r:
                print('%-34s   MDD(80%%)=%.2fpp MDD(90%%)=%.2fpp  per-fold delta: %s'
                      % ('', r['mdd_80pc'] * 100, r['mdd_90pc'] * 100,
                         ' '.join('%+.2f' % (x * 100) for x in r['per_fold_delta'])))
        print('')

    summary['verdict'] = {
        'primary_significant': [k for k, r in
                                summary['families'].get('primary', {}).get('comparisons', {}).items()
                                if r.get('significant_holm')],
        'note': 'non-significant means the difference was NOT established, '
                'not that the components are ineffective',
    }
    summary['caveats'] = [
        'n = 3 comparisons (ab_improved) cannot resolve 1-2pp effects; reported separately.',
        'Fold-level variance includes single-seed initialisation variance (seed fixed at 0).',
        'Folds differ in validation fold, so best.pt selection differs across folds.',
        'All deltas are within-fold; no cross-fold subtraction was performed.',
    ]

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('wrote %s' % out)

    if not complete:
        print('NOTE: primary models incomplete -> %s'
              % {m: len(table[m]) for m in MODELS[:4]})
    return 0


if __name__ == '__main__':
    sys.exit(main())
