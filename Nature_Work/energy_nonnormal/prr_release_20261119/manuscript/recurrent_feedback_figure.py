"""Plot the prespecified feedback runs without changing existing figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
INK, TEAL, BLUE, RUST, GRAY = '#182630', '#29776e', '#245e94', '#a0422b', '#788892'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'semibold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': INK, 'axes.edgecolor': INK,
                     'text.color': INK, 'legend.fontsize': 7,
                     'pdf.fonttype': 42})


def main():
    rec = json.loads((HERE/'recurrent_feedback_results.json').read_text())['windows']
    pair = json.loads((HERE/'feedback_allocation_results.json').read_text())['windows']
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.05), layout='constrained')
    a, b, c, d = axes.flat
    for ax, key, tag in [(a, '0.35', 'a'), (b, '1.6', 'b')]:
        group = rec[key]
        rows = list(group['active']['equal']['checkpoints'].values())[1:]
        k = [r['decisions_per_output'] for r in rows]
        med = np.array([r['share_q10_q50_q90'][1] for r in rows]).T
        for i, (y, target, color, marker) in enumerate(zip(
                med, group['target_shares'], [TEAL, RUST, BLUE, GRAY], ['o', 's', '^', 'D'])):
            ax.axhline(target, lw=.9, color=color, ls=':', alpha=.8)
            ax.plot(k, y, color=color, marker=marker, markevery=2,
                    lw=1.2, ms=3.4, label=f'Route {i+1}')
        ax.set(title=f'({tag}) Four routes, $T={key}$', ylabel='Budget share', ylim=(0, .62))
    a.legend(loc='center right', ncol=2, frameon=False, columnspacing=.9,
             bbox_to_anchor=(1, .34))

    for t, color, name in [('0.35', RUST, 'Four routes, short'), ('1.6', BLUE, 'Four routes, long')]:
        rows = list(rec[t]['active']['equal']['checkpoints'].values())[1:]
        k = [r['decisions_per_output'] for r in rows]
        for ax, field in [(c, 'tv_q10_q50_q90'), (d, 'gain_pp_q10_q50_q90')]:
            lo, med, hi = np.array([r[field] for r in rows]).T
            ax.fill_between(k, lo, hi, color=color, alpha=.13, linewidth=0)
            ax.plot(k, med, color=color, lw=1.45, label=name)
    for t, color, name in [('0.5', RUST, 'Two routes, short'), ('3.0', BLUE, 'Two routes, long')]:
        rows = pair[t]['active']['figure']
        k = np.array([int(j) for j in rows])*32
        c.plot(k, [r['absolute_share_error_quantiles_10_50_90'][1] for r in rows.values()],
               color=color, ls='--', lw=1.2, label=name)
        d.plot(k, [100*r['worst_correctness_gain_quantiles_10_50_90'][1] for r in rows.values()],
               color=color, ls='--', lw=1.2, label=name)
    c.set(title='(c) Error in budget shares', ylabel='Total variation from optimum', ylim=(0, .43))
    d.set(title='(d) Benefit to the weakest output', ylabel='Correctness gain (percentage points)', ylim=(-.21, 2.04))
    d.axhline(0, lw=.8, color=GRAY, zorder=0)
    c.legend(frameon=False, loc='upper right', labelspacing=.25, handlelength=1.9)
    for ax in axes.flat:
        ax.set(xscale='log', xlim=(110, 150000), xticks=[128, 2048, 32768, 131072],
               xlabel='Decisions per output')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.tick_params(axis='both', labelsize=7.5)
    fig.savefig(HERE/'figures'/'fig_prrS_feedback.pdf', facecolor='white')


if __name__ == '__main__':
    main()
