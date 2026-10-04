"""Plot the working recovered/empirical curves without invented uncertainty bands."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root/'south_25cm_recovered_curve.csv').open(encoding='utf-8', newline='') as f:
    curve = [r for r in csv.DictReader(f) if float(r['efficiency_fraction']) > 0]
with (root/'qg_conditioned_response_knots.csv').open(encoding='utf-8', newline='') as f:
    knots = list(csv.DictReader(f))
rows = json.loads((root/'line_response_lookup.json').read_text(encoding='utf-8'))['rows']
anchors = [r for r in rows if r['pooled_curve_anchor']]
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios':[2, 1]})
axes[0].plot([float(r['energy_keV']) for r in curve], [float(r['efficiency_fraction']) for r in curve],
             label='Recovered April 30 export (percent assumption)', color='#17629c')
axes[0].plot([float(r['energy_keV']) for r in knots], [float(r['efficiency_fraction']) for r in knots],
             'o--', ms=4, label='QG-conditioned median response (34 knots)', color='#d47a16')
axes[0].set_yscale('log')
axes[0].set_ylabel('Response / efficiency fraction')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('South HPGe, nominal 25 cm: working comparison baseline')
axes[1].scatter([r['energy_keV'] for r in anchors], [r['ratio_to_recovered_curve'] for r in anchors],
                s=16, alpha=.45, color='#d47a16')
axes[1].axhline(1, color='#17629c', lw=1)
axes[1].set_ylabel('QG response / export')
axes[1].set_xlabel('Gamma energy (keV)')
axes[1].set_xlim(50, 2000)
for ax in axes:
    ax.grid(alpha=.2)
fig.text(.12, .02, 'Working assumptions; QG agreement is not independent calibration. No uncertainty band inferred.', fontsize=9)
fig.tight_layout(rect=[0, .04, 1, 1])
fig.savefig(root/'south_efficiency_comparison.png', dpi=180)
fig.savefig(root/'south_efficiency_comparison.svg')
plt.close(fig)
