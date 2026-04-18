# Posit Connect Cloud — Redeploy Baby Steps

This guide covers redeploying the app so the existing URL
<https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/>
starts serving the A/B experiment (`app_trt.py`) instead of the old
Project-2 control app.

**Good news:** the repo is already set up so redeployment is a *single
click* on the Posit Cloud dashboard. No code edits, no settings changes,
no UI archaeology. `app.py` in the repo is now a two-line shim that
re-exports the A/B harness from `app_trt.py`, so Posit's existing
"primary file = app.py" setting still works.

---

## 0. Prereqs (30 seconds)

Verify the latest commit is actually on GitHub:

- Go to <https://github.com/ZemingLiang/STAT5243-Project3-Team21/commits/Main-Final-Deliverables>
- The top commit should be **"Redeploy prep: app.py shim re-exports A/B harness from app_trt.py"** (or a later one). If you don't see it, wait a minute and refresh — we just pushed.

---

## 1. Open the Posit Connect Cloud dashboard (1 minute)

1. In your browser, go to <https://connect.posit.cloud/>
2. Click **Log In** (top right) if you're not already signed in.
3. Sign in with the GitHub account that originally published the app (this is the same account that owns the `STAT5243-Project3-Team21` repo — most likely Zeming's).

You should land on the **Home / Content** page. If you see a list of
deployed apps, you're in the right place.

---

## 2. Find the existing deployment (30 seconds)

1. In the content list, look for a Shiny app whose URL contains
   `019d23ea-1266-cada-1d21-45e5d97e6ea5`. The title might be something
   generic like `STAT5243-Project3-Team21` or the repo name.
2. Click on its row (or title) to open the content detail page.

If you can't find it, try:
- Filtering the list by content type **Shiny**, or
- Searching for `Team21` in the content list.

---

## 3. Trigger a republish (2–3 minutes)

**This is the whole deploy step. It is one click.**

On the content detail page, look for a button/icon labeled one of:

- **Republish** (most common)
- **Redeploy**
- **Refresh from source** / a circular-arrow icon

Click it. Posit Connect Cloud will:

1. Pull the latest commit from the connected GitHub branch
   (`Main-Final-Deliverables`).
2. Install `requirements.txt` dependencies.
3. Start a fresh Shiny process using `app.py` (which is now the shim).
4. Cut traffic over to the new process once it's healthy.

You'll see a build log while this runs. Wait until it says **"Deployed"**
or the spinner stops — typically 60–180 seconds.

**Do not delete and re-publish.** That would create a new content item
with a *different* URL, and we'd have to re-share the link with everyone.

---

## 4. Verify the A/B experiment is live (1 minute)

This is the most important step. Do not skip it.

1. Open **two different incognito / private-browsing windows** (or one
   incognito window and one normal window). Incognito ensures you get a
   fresh Shiny session each time.
2. In each window, visit
   <https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/>
3. In each window, click the **Cleaning** tab in the top navigation.
4. Compare the Cleaning tab between the two windows:
   - **Group A (control):** two simple side-by-side buttons, "Preview"
     (outline dark) and "Apply" (solid dark).
   - **Group B (treatment):** a blue/white CTA box with a header
     "Step 4 — Preview, then save" and two big buttons: "Preview
     Changes" (blue) and "Apply and Save" (green). There's also a
     four-card "Guided Cleaning Workflow" at the top with steps 1–4.

If **both** windows show the same layout, keep refreshing one of them
(close the tab, open a new incognito tab, revisit the URL) until you
see the other layout. The assignment is 50/50 random per session, so
you should hit both arms within ~3–4 tries.

**What you should NOT see anywhere on the page:**
- The text "Version A / Control"
- The text "Version B / Guided"
- Any explicit mention of "A/B test", "experiment", "treatment", or
  "control" in user-facing copy (except the small consent notice on
  the Guide tab's Welcome card).

If you see any of those strings, you're on an older deployment — go
back to step 3 and click republish again.

---

## 5. Test the event logger (30 seconds)

Pick one of the incognito windows and do this:

1. Go to the **Load** tab → click **Load Sleep, Mobile and Stress**.
2. Go to the **Cleaning** tab.
3. Pick an action — e.g., **Handle missing values** with strategy **k-NN
   imputation**.
4. Select a column (e.g., `age`).
5. Click **Preview** (or **Preview Changes**).
6. Click **Apply** (or **Apply and Save**).

This should log four events to `ab_test_events.csv` on the Posit Cloud
server (one preview + one apply per click, plus any resolution event).

---

## 6. Test the admin download (1 minute)

Still in the same incognito window:

1. Go to the **Guide** tab.
2. Scroll to the bottom → expand **"Team only — download A/B event log"**.
3. Enter the password: **`team21-cleaning-ab`**
4. A button labeled **"Download ab_test_events.csv"** should appear.
5. Click it. The file should download to your laptop.
6. Open the file in a text editor or Excel — you should see 3–5 rows,
   one per click you made in step 5, including the `ab_group` column
   showing `A` or `B`.

If the password works and the file downloads cleanly, you're **done**.
The experiment is live.

---

## 7. Share the link

Once verification passes, the link to share everywhere is:

```
https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/
```

Suggested sharing copy (paste into Slack / Reddit / LinkedIn / WeChat):

> I built a free no-code web tool for cleaning and exploring datasets —
> upload a CSV, fix missing values, make new columns, run EDA, all in
> the browser. Would love 2 minutes of feedback if you try it out:
> https://019d23ea-1266-cada-1d21-45e5d97e6ea5.share.connect.posit.cloud/

(Don't mention "A/B test" in the sharing copy — keep users blind.)

---

## During the collection window

- Pull the log every few hours via the admin download, just in case the
  Posit container restarts. Replace the local file each time with the
  latest version.
- After cutoff (2026-04-19 ~12:00 EST), run the analysis + report-fill
  pipeline:

```bash
python ab_analysis.py ab_test_events.csv --out figures/ \
       --fill-template REPORT.md --fill-out REPORT.filled.md
pandoc REPORT.filled.md -o report.pdf --pdf-engine=xelatex \
       -V mainfont="Helvetica Neue" -V monofont=Menlo --toc --toc-depth=2
```

Commit + push final. Submit.

---

## Troubleshooting

**Republish fails with a build error.**
Click the failed build to see the log. 99% of the time it's a missing
dependency — check `requirements.txt` has every package used by
`app_trt.py`. Our `requirements.txt` currently covers everything; if
the build still fails, compare the build log's "ModuleNotFoundError"
line against `requirements.txt`.

**Can't find the Republish button.**
The UI may show it as an icon only (a circular arrow or refresh symbol)
rather than a labeled button. Hover over icons on the content detail
page to find it. If still stuck, the fallback is the "Content Settings"
menu → "Git" tab → "Sync / Rebuild now".

**The deployed URL is down or returns 503.**
Wait a minute and refresh. Posit's free tier takes ~20 seconds to spin
up a cold container after idle.

**Both incognito windows show the same Cleaning layout forever.**
Random chance can give you two-of-the-same for a few refreshes. Try
5–6 times. If all 6 are identical, something's wrong with the
random assignment — grep the server logs for a Python traceback.

**Admin download button never appears even with the right password.**
The log file doesn't exist yet — you need to fire at least one
preview/apply event on the Cleaning tab first (step 5 above). The
button shows up only after the CSV exists on disk.
