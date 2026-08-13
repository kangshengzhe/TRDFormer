"""
Regenerate every manuscript figure, in the order they appear in the paper.

One entry point so the figure set can never drift out of sync with the data:
run this after any change to the results tree and all seven in-article
figures, plus the graphical abstract, are rebuilt from the same records, with
the same design system, at print size.

Each builder also writes a downscaled ``_preview/`` copy; the print assets
are 640 dpi and exceed 2000px on the long edge, which some image consumers
refuse, so review the previews.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "visualization"))
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_all")

#: (module, figure number, what it replaced) - the third field is the record
#: of which legacy assets each new figure subsumes.
FIGURES = [
    ("fig_data_motivation", "1",
     "fig01_workflow_overview + fig_feature_correlation"),
    ("fig_mechanism_strip", "3",
     "fig03_vmd_decomposition (companion to the TikZ architecture)"),
    ("fig_main_comparison", "4",
     "fig_overall_comparison + fig_performance_overview"),
    ("fig_prediction_matrix", "5",
     "fig_panels_h1/h6/h24 + fig_prediction_panels_main + fig_qualitative_4h"),
    ("fig_ablation_gate", "6",
     "fig_ablation_bars + fig_branch_contribution"),
    ("fig_ramp_analysis", "7", "fig_ramp_window_analysis"),
    ("fig_generalization", "8",
     "fig_multiturb_generalization + fig_temporal_robustness"),
    # Not an in-article figure: a separate submission item, written to
    # manuscript/figures/graphical_abstract.{pdf,png} at the 13 x 5.2 cm /
    # 1328:531 size the journal specifies. Built here so it cannot drift out
    # of sync with the numbers it quotes.
    ("fig_graphical_abstract", "GA", "fig00_graphical_abstract (schematic)"),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(ROOT)
    failed = []
    for mod_name, num, replaces in FIGURES:
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            rc = mod.build()
            status = "ok" if rc == 0 else f"rc={rc}"
        except Exception as exc:                     # keep going, report at end
            logger.exception("Fig. %s (%s) failed", num, mod_name)
            status = f"FAILED: {exc}"
            rc = 1
        if rc != 0:
            failed.append((num, mod_name, status))
        logger.info("Fig. %-3s %-24s %-8s %5.1fs   <- %s",
                    num, mod_name, status, time.time() - t0, replaces)

    if failed:
        logger.error("%d figure(s) failed: %s", len(failed),
                     ", ".join(f"Fig.{n}" for n, _, _ in failed))
        return 1
    logger.info("all %d figures rebuilt", len(FIGURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
