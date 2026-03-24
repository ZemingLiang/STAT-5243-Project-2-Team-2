# STAT-5243-Project-2-Team-2

Group members: Zeming Liang (`zl3688`), Yuhan Guo (`yg2695`), Baixuan Chen (`bc3212`), Cecilia Zang (`cz2957`)

## Current App Architecture

This branch implements the project as a single local `Shiny for Python` app.

- `app.py`: Shiny app entrypoint and UI/server logic
- `EDA.py`: direct summary, filtering, and EDA logic
- `p2_divided.py`: direct cleaning and preprocessing logic
- `feature_engineering.py`: pure DataFrame-to-DataFrame feature engineering logic
- `test_data/sleep_mobile_stress_dataset_15000.csv`: built-in local dataset
- `shiny_local_smoke_test.py`: direct local integration smoke test

There is no Flask app in the run path and no repo-owned REST API in the new architecture. The UI imports and calls local Python modules directly.

## Run The App

```bash
shiny run app.py
```

Then open `http://127.0.0.1:8000`.

## Smoke Test

```bash
python3 shiny_local_smoke_test.py
```

## Supported v1 Features

- Built-in dataset loading
- Upload from `CSV`, `Excel`, and `JSON`
- Cleaning and preprocessing with preview-first workflow
- Feature engineering with preview-first workflow
- Active dataset version switching with local in-memory history
- EDA summaries, filtering, 1D plots, 2D plots, regression, and multiline plots

## Notes

- `RDS` is intentionally out of scope for v1
- Derived dataset history is stored only in memory for the current app session
- The final app is meant to satisfy the project PDF directly rather than the older Flask/API prototype direction
