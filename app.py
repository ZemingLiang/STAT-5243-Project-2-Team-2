"""Posit Connect Cloud entry point for STAT 5243 Project 3.

Posit Connect Cloud was originally configured with ``app.py`` as the
"primary file" for this deployment. Rather than changing that setting in
the Posit dashboard, this module re-exports the A/B-test Shiny app from
``app_trt.py`` so the existing deployment URL serves the experiment
without any configuration change — a ``republish`` click on the Posit
dashboard is the only action required after we push.

The original Project-2 control app (2486 lines) was renamed to
``app_project2_reference.py`` on the same commit; git history prior to
that commit also preserves it. If you want to run the pure Project-2
control locally: ``shiny run app_project2_reference.py``.
"""

from app_trt import app, load_builtin_dataset  # noqa: F401
