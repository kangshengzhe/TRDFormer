"""
Visualization suite for the wind power forecasting paper.

Current entry point: run ``python -m visualization.build_all_figures`` to
regenerate every manuscript figure (Fig. 1, 3-8) from the same records, with
the same design system (``_style.py``/``_data.py``), at print size. See that
module's docstring for the per-figure module list.

2026-08 cleanup note: this package previously re-exported
``plot_prediction_curve``/``plot_error_distribution`` at import time. Both
modules were superseded by the ``fig_*.py`` builders above and moved to
``_cleanup_archive/visualization_legacy/``. A couple of older orchestration
scripts (``scripts/aggregate_cli.py``, ``scripts/update_paper_v2.py``) still
import them directly inside try/except blocks; those calls will now raise
ImportError and be caught there, which only disables their (already
superseded) prediction-curve/error-distribution output -- it does not affect
``build_all_figures.py`` or anything the paper currently uses.
"""

__all__: list[str] = []
