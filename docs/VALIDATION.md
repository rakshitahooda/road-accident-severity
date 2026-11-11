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
