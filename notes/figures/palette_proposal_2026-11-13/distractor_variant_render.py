"""Render Fig. 1 (v4), Fig. 4 (HP blocks) and Fig. 6 (v13) with one distractor-colour variant, into a scratch folder.

    python distractor_variant_render.py {sky,green} OUT_DIR      (run from the repo root, n = 28 via run_excluding)

sky:   distractor #4AA3DF, pRF dark teal #00796B, decoding maps purple <-> green (the palette-A proposal)
green: distractor bluish green #009E73 (Okabe-Ito), pRF charcoal #3E4C59, decoding maps purple <-> gold
The paper_style constants are patched before the figure modules are imported; nothing in the repo is written.
"""
import sys
from pathlib import Path

import retsupp.visualize.paper.paper_style as ps

variant, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ps.OUT = out
if variant == 'green':
    ps.DISTRACTOR = '#009E73'
    ps.DISTRACTOR_CMAP = ps._cmap('distractor', ['#FFFFFF', ps.DISTRACTOR])
    ps.PRF = '#3E4C59'
    ps.DIV_MORE = '#8C6D1F'
    ps.DIV_CMAP = ps._cmap('less_more', [ps.HP_DEEP, ps.HP, '#C2A5CF', '#F7F7F7', '#F0D98C', '#D4A92A', ps.DIV_MORE])
import retsupp.visualize.paper.run_excluding as rx  # noqa: E402

steps = [['retsupp.visualize.paper.fig1_paradigm_behaviour_v4', '--exclude-subjects', '1', '2', '--suffix', f'_{variant}'],
         ['retsupp.visualize.paper.fig4_model', '--version', 'blocks', '--name', f'fig4_{variant}'],
         ['retsupp.visualize.paper.fig6_decoding', 'plot', '--v12', '--name', f'fig6_{variant}']]
for st in steps:
    sys.argv = ['run_excluding', '--exclude', '1', '2', '--'] + st
    try:
        rx.main() if hasattr(rx, 'main') else None
    except SystemExit:
        pass
