# EDA Slide Deck Outline — STAT 5243 Project 3

Target: ~7 minutes, 8 slides. Paste into Google Slides or Keynote; images live in `figures/`.

---

## Slide 1 — Title

- **STAT 5243 Project 3**
- **A/B Test on Data-Cleaning UX**
- Bohong Zheng · Zeming Liang · Zuer Weng · Maya Rubin
- April 2026
- Columbia University

---

## Slide 2 — Motivation

- Project 2: we built an interactive Shiny-for-Python data workbench (Load → Overview → Cleaning → Feature Engineering → EDA).
- Usability friction: new users click **Apply** without **Preview**, overwrite the dataset, abandon the session.
- The Cleaning tab has nine operations — the most configurable part of the app, and the most painful UX.

**One sentence for the audience:** "Can a small UI nudge change how cautiously users clean their data?"

---

## Slide 3 — Research Question and Hypothesis

- **RQ:** Does a guided four-step workflow layout with a prominent "Preview, then Save" CTA improve users' preview-before-apply behaviour on the Cleaning tab?
- **H0:** The guided layout (Version B) does not change the session-level apply_rate.
- **H1:** The guided layout changes apply_rate in either direction.
- Secondary metrics (Bonferroni-corrected, n=4): preview_to_apply_conversion, apply_success_rate, successful_actions_per_session.

---

## Slide 4 — Experimental Design

- **Single deployed app** (`app_trt.py`) — same URL for every user.
- On session start: `ab_group = random.choice(["A","B"])`.
- Conditionally renders the Cleaning tab's sidebar + action buttons. Everything else identical.
- Every preview/apply click logged to `ab_test_events.csv`.
- **Side-by-side screenshot** — Version A (flat buttons) vs. Version B (guided CTA box with "Step 4 — Preview, then save").

---

## Slide 5 — Data Collection

- **Window:** 2026-04-18 09:30 EST → 2026-04-19 12:00 EST (≈ 26 hours).
- **Channels:** Course Slack, LinkedIn, WeChat, Reddit r/datascience, personal messages.
- **Observed:** *N_A* control sessions, *N_B* treatment sessions. (fill in from `ab_analysis.py` output)
- **Schema:** timestamp, session_id, ab_group, event_type, clean_action, dataset_key, columns_count, success, seconds_since_session_start, details.
- **No PII:** only uuid session IDs.

---

## Slide 6 — Results (key figure)

- Insert **`figures/apply_rate_by_group.png`** (per-session apply_rate histogram by group).
- Put the results table as a small overlay or on the next slide.

**Talking points:**
- A's distribution clusters at high apply_rate (many "apply without preview" sessions).
- B's distribution shifts left (more previews before applies).
- Cohen's d = *{d_1}*; Bonferroni-adjusted p = *{p_1_adj}*.

---

## Slide 7 — Results (full table) and Interpretation

| Metric | Mean A | Mean B | p (Bonferroni) | Decision at α = 0.05 |
|---|---|---|---|---|
| apply_rate (primary) | *{mu_A_1}* | *{mu_B_1}* | *{p_1_adj}* | *{dec_1}* |
| preview_to_apply_conversion | *{mu_A_2}* | *{mu_B_2}* | *{p_2_adj}* | *{dec_2}* |
| apply_success_rate | *{mu_A_3}* | *{mu_B_3}* | *{p_3_adj}* | *{dec_3}* |
| successful_actions_per_session | *{mu_A_4}* | *{mu_B_4}* | *{p_4_adj}* | *{dec_4}* |

**Plain-English interpretation:** *{one-sentence summary of what the data says}*.

---

## Slide 8 — Limitations and Next Steps

- **Randomisation is session-scoped**, not user-scoped — a returning user may flip groups.
- **Small, self-selected sample** — classmates and personal networks are more technical than the general user base.
- **First-exposure effects only** — no long-run learning / retention captured.
- **Cleaning tab only** — five other tabs were not part of the experiment.
- **Next steps:** user-scoped randomisation via cookie; longer collection window; A/A control arm; expand instrumentation to Feature Engineering and EDA tabs.

---

## Delivery notes for Zeming

- 7 minutes ≈ ~50–60 seconds per slide. Trim Slide 4 if time is tight (merge with Slide 5).
- Pull numbers from `ab_analysis.py`'s printed summary table; rerun on the frozen log at the 2026-04-19 12:00 cutoff.
- Figures are in `figures/` — regenerate with `python ab_analysis.py ab_test_events.csv --out figures/` after the freeze.
- Screenshots for Slide 4: take them from the deployed app at the shared URL (open in two incognito windows to force different group assignments).
