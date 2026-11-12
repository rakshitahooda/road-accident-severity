# Validation record

Rechecked locally on 2026-11-07 after adding profile-cluster bootstrap evaluation:

- Reproducible training completed against the reviewed dataset checksum; the balanced random forest remained selected, with test macro-F1 0.442. The new 300-resample profile-cluster sensitivity interval was 0.411–0.473 over 2,258 test profiles; no resample omitted a class.
- 20 pytest checks passed, including the new cluster-bootstrap reproducibility and input-validation checks.
- Ruff checks passed for `roadlab` and `tests`.
- Browser/UI smoke checks were not rerun for this change; the results below are the earlier 2026-11-05 run.

Executed locally on 2026-11-05 using Python 3.11, in a newly created environment with requirements-dev.txt:

- Dataset checksum matched the reviewed 12,316-row snapshot; all four candidates were retrained.
- 18 pytest checks passed: live API inference and batch equivalence, schema/range errors, null handling, batch limits, split audit, missing-artifact service failures, changed-dataset rejection and OpenAPI bounds.
- Ruff checks passed for application, tests and browser smoke script.
- Playwright Chromium: real inference HTTP request succeeded; evidence/method/experiment views worked; four comparison rows rendered; no JavaScript errors; no horizontal overflow at 390px. Screenshots are in this folder.
- Notebook JSON and code-cell Python syntax validated. Notebook not executed separately; it invokes the trained/tested modules.

To repeat browser checks, install Playwright in a development environment (`pip install playwright==1.51.0; playwright install chromium`), start the documented local API, then run `python scripts/check_ui.py`. Browser dependencies are optional, separate from runtime requirements.

This is a local record, not a claim that GitHub Actions or production hosting passed. Test output includes upstream dependency deprecation warnings, not failed assertions.

## Rechecked locally on 2026-11-09 after adding grouped development cross-validation

- Reproducible training completed against the reviewed dataset checksum. Five-fold `StratifiedGroupKFold` used 9,823 development rows and 127 fatal-labelled rows; identical input-profile groups had zero overlap in every fold. The fixed 2,493-row test assignment was verified against the unchanged seeded split and excluded from all CV folds.
- Out-of-fold fatal recall / macro-F1: majority 0/127 (0.000) / 0.305; balanced logistic regression 63/127 (0.496) / 0.282; balanced decision tree 58/127 (0.457) / 0.365; balanced random forest 27/127 (0.213) / 0.425. The forest leads on macro-F1; logistic regression catches the most fatal rows. This is exploratory, since the previous validation split selected the current model. Identical profiles are not incident IDs.
- 26 pytest checks passed. Ruff checks, JavaScript syntax validation and `git diff --check` passed.
- Playwright Chromium smoke check passed live inference, keyboard tabs, fatal-audit and grouped-CV rendering, and the Evidence tab; no JavaScript errors or horizontal overflow at 390px. The refreshed Evidence screenshot is `docs/evidence.png`.
- These are local checks, not a claim that GitHub Actions or production hosting passed. Dependency deprecation warnings remain. Changes are still local and have not been committed or pushed.


## Rechecked locally on 2026-11-11 after adding grouped out-of-fold fatal-score exploration

- The additional cutoff curves use only out-of-fold predictions from the five grouped development folds. They report fatal recall alongside the fraction of nonfatal rows flagged, without selecting or recommending a cutoff; raw model scores are uncalibrated and the live three-class inference path is unchanged.
- Reproducible training completed on the reviewed dataset. The balanced random forest remains selected; held-out test macro-F1 is 0.442 and the profile-cluster bootstrap interval is 0.411–0.473. The test assignment and live three-class model are unchanged by this analysis.
- The grouped out-of-fold cutoff explorer is descriptive only. For logistic regression at 0.50, it flags 36/127 fatal-labelled development rows and 1,198/9,696 nonfatal rows. These uncalibrated scores are not an operating recommendation; lowering the threshold changes both counts.
- 32 pytest checks passed. Ruff checks, JavaScript syntax validation and `git diff --check` passed. Playwright Chromium passed live inference and tab interactions; the 390px evidence view had no horizontal overflow or JavaScript errors.
- The threshold-explorer evidence image was refreshed. These are local checks, not a claim that GitHub Actions or production hosting passed.


The fixed test set contains 31 fatal-labelled rows, of which 11 were classified as fatal (recall 0.355). The 95% Wilson interval is 0.211–0.531; it reflects binomial sampling variation for this fixed model and holdout only, not uncertainty from retraining, model selection, or new datasets.

## Rechecked locally on 2026-11-12 after adding grouped out-of-fold fatal-score reliability diagnostics

- Added fixed-width reliability bins and Brier scores for each candidate's raw one-vs-rest fatal scores across the grouped development folds (9,823 rows, including 127 fatal-labelled rows). No calibrator is fitted, and these summaries are not used for model or cutoff selection; the held-out test set and live three-class inference path remain unchanged.
- 40 pytest checks passed. Ruff, JavaScript syntax, Python compilation and `git diff --check` passed.
- Playwright Chromium passed live inference, tab interaction and the fatal-score reliability display, with no JavaScript errors or horizontal overflow at 390px.
- These are local checks, not a claim that GitHub Actions or production hosting passed. The scores remain uncalibrated and the project is not a safety tool.

## Cross-version checks and CI coverage on 2026-11-12

- Repeated model training with Python 3.12.14; the balanced random forest remained selected and the 11/31 held-out fatal-case result remained unchanged. The full test suite passed: 40 checks.
- GitHub Actions is configured to train and test on Python 3.11 and 3.12, run Ruff, and check frontend JavaScript syntax with Node 22. This local record does not claim a hosted Actions run has passed.
