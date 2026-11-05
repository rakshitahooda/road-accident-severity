# Validation record

Executed locally on 2026-11-05 using Python 3.11, in a newly created environment with requirements-dev.txt:

- Dataset checksum matched the reviewed 12,316-row snapshot; all four candidates were retrained.
- 18 pytest checks passed: live API inference and batch equivalence, schema/range errors, null handling, batch limits, split audit, missing-artifact service failures, changed-dataset rejection and OpenAPI bounds.
- Ruff checks passed for application, tests and browser smoke script.
- Playwright Chromium: real inference HTTP request succeeded; evidence/method/experiment views worked; four comparison rows rendered; no JavaScript errors; no horizontal overflow at 390px. Screenshots are in this folder.
- Notebook JSON and code-cell Python syntax validated. Notebook not executed separately; it invokes the trained/tested modules.

To repeat browser checks, install Playwright in a development environment (`pip install playwright==1.51.0; playwright install chromium`), start the documented local API, then run `python scripts/check_ui.py`. Browser dependencies are optional, separate from runtime requirements.

This is a local record, not a claim that GitHub Actions or production hosting passed. Test output includes upstream dependency deprecation warnings, not failed assertions.
