"""Plot latency using the exact values in the report table."""
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    lines = Path(sys.argv[1]).read_text(encoding='utf-8').splitlines()
    rows = [[cell.strip() for cell in line.rstrip('%').removesuffix(r'\\').split('&')]
            for line in lines if re.match(r'^\s*\d+\s*&', line)]
    batches = [int(row[0]) for row in rows]
    if batches != [8, 16, 32, 64, 128] or len({len(r) for r in rows})!=1 or len(rows[0]) not in (5,9):
        raise ValueError('Expected five batch-size rows')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.65), layout='constrained')
    panels = [(3, 'Latency per image', 'ms/image'),
              (2, 'Latency per batch', 'ms/batch')]
    for ax, (col, title, unit) in zip(axes, panels):
        series = [(0,'Router','#187F80','o')]
        if len(rows[0])==9:
            series.append((4,'ResNet101 FP16','#183B56','s'))
        values = []
        for offset,label,color,marker in series:
            vals = [float(row[col+offset]) if row[col+offset] != r'\missing' else float('nan') for row in rows]
            ax.plot(batches,vals,color=color,marker=marker,label=label,lw=1.6,ms=4)
            values.extend(v for v in vals if v==v)
        if len(series)==2:
            ax.legend(frameon=False,fontsize=8)
        ax.set(title=title, xlabel='Batch size', ylabel=unit, ylim=(0, 1.15*max(values)), xlim=(1, 137))
        ax.set_xticks(batches)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(sys.argv[2], dpi=220)


if __name__ == '__main__':
    main()
