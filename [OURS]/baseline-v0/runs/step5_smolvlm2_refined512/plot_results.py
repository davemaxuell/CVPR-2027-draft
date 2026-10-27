"""Plot completed matched trials without changing any training artifact."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    result = read(RUN / 'result.json')
    assert result['status'] == 'complete'
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for trial, label, color in [('transfer', 'PointMaze transfer', '#117A65'),
                                ('fresh', 'Fresh dynamics', '#3B61A8')]:
        rows = [json.loads(line) for line in (RUN / trial / 'train.jsonl').read_text().splitlines()]
        # A resumed segment can repeat logged steps after its last durable checkpoint.
        rows = sorted({r['step']: r for r in rows}.values(), key=lambda r: r['step'])
        axes[0].plot([r['samples_seen'] / 1e6 for r in rows], [r['train_mse'] for r in rows], color=color, label=label, linewidth=1.4)
        evaluated = sorted([read(p) for p in (RUN / trial / 'evaluations').glob('step_*.json')], key=lambda r: r['step'])
        axes[1].plot([r['seen'] / 1e6 for r in evaluated], [r['macro_mse'] for r in evaluated], marker='o', markersize=4, linewidth=1.7, color=color, label=label)
        best = read(RUN / trial / 'best.json')
        axes[1].scatter([best['seen'] / 1e6], [best['macro_mse']], marker='*', s=150, color=color, edgecolors='black', linewidths=.6, zorder=5)
    baseline = result['trials']['transfer']['metrics']['copy_current_macro_mse']
    axes[1].axhline(baseline, color='#777777', linestyle='--', linewidth=1, label='Copy current features')
    axes[0].set_title('Training loss (100-update means)')
    axes[1].set_title('Held-out group-C prediction error')
    for ax in axes:
        ax.set_xlabel('Training transitions seen (millions)')
        ax.set_ylabel('Raw-patch MSE (log scale)')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which='both', alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('DINO dynamics: matched one-pass comparison', fontsize=13)
    fig.savefig(RUN / 'learning_curves.png', dpi=180)
    fig.savefig(RUN / 'learning_curves.svg')
    plt.close(fig)
    print('Saved learning_curves.png and learning_curves.svg')


if __name__ == '__main__':
    main()
