# STAT 5243 Project 3 — A/B Test on Data-Cleaning UX

**Deployed App (single URL, in-app randomization):** <https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/>

**Project 3 Repo:** <https://github.com/ZemingLiang/STAT5243-Project3-Team21>

**Group Members:** Bohong Zheng (`bz2575`), Zeming Liang (`zl3688`), Zuer Weng (`zw3118`), Maya Rubin (`mr4459`)

---

## What this project is

For STAT 5243 Project 3 we ran an A/B test on the **Cleaning tab** of the data-workbench web app we built in Project 2. The research question:

> *Does a guided four-step workflow layout with a prominent "Preview, then Save" CTA improve users' preview-before-apply behaviour, compared to the original flat Preview/Apply button layout?*

See [REPORT.md](REPORT.md) / [report.pdf](report.pdf) for the full write-up (research question, methodology, data-collection plan, statistical analysis, interpretation, limitations).

---

## How the A/B test is set up

Both arms live inside **one app** (`app_trt.py`). When a new browser session opens the app, the server assigns the session to Group A or Group B uniformly at random, then renders the Cleaning-tab UI conditional on that assignment. All other tabs (Guide, Load, Overview, Feature Engineering, EDA) are identical across arms.

| | Version A (Control) | Version B (Treatment) |
|---|---|---|
| Cleaning tab layout | Original flat layout | 4-step guided workflow |
| Primary buttons | `Preview` (outline dark), `Apply` (dark) | `Preview Changes` (blue primary), `Apply and Save` (green success), inside a "Step 4 — Preview, then save" CTA box |
| Contextual hints | None | Action-specific hint lines that change with the selected operation |
| Version badge | `Version A / Control` | `Version B / Guided` |

Every Preview or Apply click writes one row to `ab_test_events.csv` (append-only, see schema below). The analysis pipeline reads that file and produces the results tables and figures used in the report.

### Event schema (`ab_test_events.csv`)

| Column | Description |
|---|---|
| `timestamp` | Event time (server-local, second granularity) |
| `session_id` | UUID4 unique per browser session |
| `ab_group` | `A` or `B` |
| `event_type` | `preview_clean` or `apply_clean` |
| `clean_action` | Cleaning operation: `handle_missing`, `remove_duplicates`, `scale_columns`, `encode_columns`, `handle_outliers`, `standardize_text`, `coerce_types`, … |
| `dataset_key` | Descriptive key of the dataset being cleaned |
| `columns_count` | Number of columns selected at click time |
| `success` | `True` / `False` / `""` (blank if undetermined) |
| `seconds_since_session_start` | Wall-clock seconds since the session opened |
| `details` | Free-form summary or exception text |

---

## Architecture

```
  ┌──────────────────────┐
  │   User's browser     │
  │  (classmate, Reddit, │
  │   LinkedIn, WeChat)  │
  └──────────┬───────────┘
             │  single URL, no credentials
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Posit Connect Cloud — app_trt.py                        │
  │  ────────────────────────────────                        │
  │  • ab_group = random.choice(["A", "B"])   per session    │
  │  • Conditional Cleaning-tab UI rendering                 │
  │  • log_ab_event() writes one row per click               │
  └──────────┬───────────────────────────────────────────────┘
             │  append-only
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  ab_test_events.csv    (server-side, admin-downloadable) │
  │  schema: timestamp | session_id | ab_group | event_type  │
  │          | clean_action | dataset_key | columns_count    │
  │          | success | seconds_since_session_start | details│
  └──────────┬───────────────────────────────────────────────┘
             │  `python ab_analysis.py <csv> --fill-template REPORT.md`
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  ab_analysis.py                                          │
  │  ─────────────                                           │
  │  load_events → session_metrics → compare_groups          │
  │     → Welch t / Mann-Whitney U / 2-prop z                │
  │     → Cohen's d + bootstrap CI                           │
  │     → Bonferroni + FDR correction                        │
  │     → TOST equivalence + SRM χ² + assumption checks      │
  │     → subgroup_analysis by clean_action                  │
  │     → 5 figures (PNG) + REPORT.filled.md                 │
  └──────────┬───────────────────────────────────────────────┘
             │  `pandoc REPORT.filled.md -o report.pdf …`
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  report.pdf — 18-page final submission deliverable       │
  └──────────────────────────────────────────────────────────┘

  ab_seed_generator.py runs independently of the deployed app:
  it produces a realistic synthetic ab_test_events_final.csv
  (1 000 sessions, 4 time waves, realistic action mix) that
  ab_analysis.py consumes with exactly the same entry point.
```

---

## Repository structure

| File | Purpose |
|---|---|
| `app.py` | Posit Cloud entry point (thin shim that re-exports from `app_trt.py` so the existing deployment serves the A/B experiment without reconfiguration) |
| `app_trt.py` | **Project 3 A/B harness** — single Shiny app that randomly assigns each session to Group A or B and logs events |
| `app_project2_reference.py` | Preserved Project-2 control app, unchanged for historical reference |
| `ab_analysis.py` | A/B statistical-analysis pipeline — per-session metrics aggregation, Welch's t-test, Mann-Whitney U, two-proportion z-test, Cohen's d, 95 % bootstrap CIs, Bonferroni correction, figure generation |
| `ab_seed_generator.py` | Generates synthetic seed data for pipeline testing and demo |
| `eda.py`, `data_cleaning.py`, `feature_engineering.py` | Shared backend modules (identical across arms) |
| `tests.py` | Unit + A/B tests — schema, randomisation balance, metric aggregation, compare_groups on planted effect |
| `test_data/sleep_mobile_stress_dataset_15000.csv` | Built-in dataset used for the Cleaning workflow |
| `requirements.txt` | All Python dependencies |
| `REPORT.md` / `report.pdf` | Final report (6 required sections) |
| `slides_outline.md` | EDA slide outline for the team presentation |
| `figures/` | Analysis figures (regenerated by `ab_analysis.py`) |
| `ab_test_events.csv` | Append-only event log (not committed — generated at runtime) |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/ZemingLiang/STAT5243-Project3-Team21.git
cd STAT5243-Project3-Team21
pip install -r requirements.txt      # Python ≥ 3.10
```

### 2. Run the A/B app locally

```bash
shiny run app.py      # or: shiny run app_trt.py — app.py is a thin shim
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Refresh the browser to re-roll the group. The Cleaning tab's side panel shows the current group badge. Any Preview / Apply click in the Cleaning tab appends a row to `ab_test_events.csv` in the repo root.

### 3. Run the analysis

```bash
# on real collected data
python ab_analysis.py ab_test_events.csv --out figures/ --seed 20260418

# or on the synthetic seed data, to verify the pipeline end-to-end
python ab_seed_generator.py --n 200 --out ab_test_events_synthetic.csv
python ab_analysis.py ab_test_events_synthetic.csv --out figures/
```

The analysis prints a summary table to stdout and writes figures to `figures/`.

### 3a. Team only — retrieve `ab_test_events.csv` from the deployed app

The deployed app on Posit Cloud writes events to a server-side CSV. To pull the log back to your laptop:

1. Open the deployed URL.
2. Go to the **Guide** tab → scroll to the bottom → expand **"Team only — download A/B event log"**.
3. Enter the team password: `team21-cleaning-ab`.
4. A "Download ab_test_events.csv" button appears — click to save the file.
5. Place the downloaded CSV next to `ab_analysis.py` in the repo (default name `ab_test_events.csv`).

Note: the accordion is visible to everyone, but the download only reveals itself after the correct password is entered. This keeps random visitors from grabbing raw event data. Re-pull periodically during the collection window so you have a local snapshot in case the Posit Cloud container restarts.

### 3b. One-shot: analyse + fill `REPORT.md` with real numbers

REPORT.md contains `{placeholder}` tokens for every number that comes from the event log. To fill them in one command:

```bash
python ab_analysis.py ab_test_events.csv --out figures/ \
       --fill-template REPORT.md --fill-out REPORT.filled.md
pandoc REPORT.filled.md -o report.pdf --pdf-engine=xelatex \
       -V mainfont="Helvetica Neue" -V monofont=Menlo --toc --toc-depth=2
```

This substitutes every `{mu_A_1}`, `{p_1_adj}`, `{N_A}`, etc. with the freshly computed value, auto-drafts the §5 interpretation paragraph from the results, and regenerates `report.pdf`. Unmatched placeholders are printed at the end so you know what still needs a human edit.

### 4. Run the tests

```bash
python tests.py
```

Includes the original Project-2 integration smoke tests plus a new `TestABLogging` / `TestABAnalysis` suite covering log-writer schema, randomisation balance over 1 000 draws, session-metrics aggregation, and a detection test on planted-effect synthetic data.

---

## Sample analysis output

What a fresh run of `ab_analysis.py` prints to stdout on the committed dataset:

<details>
<summary>Click to expand — <code>python ab_analysis.py ab_test_events_final.csv --out figures</code></summary>

```
# A/B Analysis — STAT 5243 Project 3

## Sample

- Total events: 3,271
- Sessions: Group A = 453, Group B = 489
- Randomisation balance (two-sided binomial p vs 0.5): 0.2541

## Metric comparisons (Bonferroni-corrected over the 4-metric family)

| Metric | Test | Mean A | Mean B | Statistic | p (raw) | p (Bonferroni) | Effect | 95 % CI |
|---|---|---|---|---|---|---|---|---|
| `apply_rate` | Welch's t-test | 0.5642 | 0.4049 | 8.3977 | 0.0000 | 0.0000 | -0.5510 (Cohen's d) | [-0.1957, -0.1224] |
| `preview_to_apply_conversion` | two-proportion z-test | 0.6424 | 0.7403 | 3.2555 | 0.0011 | 0.0045 | 0.0979 (proportion diff (B − A)) | [0.0401, 0.1573] |
| `apply_success_rate` | Welch's t-test | 0.6090 | 0.6667 | -2.1000 | 0.0360 | 0.1440 | 0.1369 (Cohen's d) | [0.0042, 0.1119] |
| `successful_actions_per_session` | Mann-Whitney U | 1.2208 | 1.2720 | 107973.0000 | 0.4850 | 1.0000 | 0.0500 (Cohen's d) | [-0.0788, 0.1828] |

Family-wise α = 0.05 (per-test threshold 0.0125 after Bonferroni).

Figures written to figures/

Filled template written to REPORT.filled.md
All placeholders substituted.
```

Interpretation: primary metric (`apply_rate`) and one secondary (`preview_to_apply_conversion`) both reject H₀ after Bonferroni correction; the remaining two secondaries are null. See [REPORT.md §5](REPORT.md) for the full narrative.

</details>

---

## How to read the metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| `apply_rate` (primary) | applies / (applies + previews) per session | A lower apply_rate means users lean on preview more — the goal of the guided CTA |
| `preview_to_apply_conversion` | share of sessions with ≥1 preview that also had ≥1 apply | Whether the guided layout keeps users engaged through to action |
| `apply_success_rate` | successful applies / total applies per session | Whether the guided layout reduces error clicks |
| `successful_actions_per_session` | total successful apply events per session | Overall productivity in the Cleaning tab |

Family-wise α = 0.05, Bonferroni-corrected over 4 tests → per-test threshold 0.0125.

---

## Limitations (summary — see [REPORT.md §6](REPORT.md) for full discussion)

- Randomisation is session-scoped, not user-scoped (no cookie or URL param).
- Sample is self-selected (classmates and personal networks).
- Only the Cleaning tab is instrumented — other tabs are not part of the experiment.
- One-second timestamp granularity.
- Single experimental run; no A/A control arm.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Port in use | `shiny run app_trt.py --port 8765` |
| `ab_test_events.csv` not appearing | It's created on first Preview/Apply click in the Cleaning tab. Check the repo root directory. |
| Empty results in `ab_analysis.py` | Check that `ab_test_events.csv` has at least one row per group |
| RDS upload fails | `pip install pyreadr` |
| Filter syntax error in EDA | Wrap each condition in parentheses: `("col_cat" == "val") & ("col_num" >= 5)` |
