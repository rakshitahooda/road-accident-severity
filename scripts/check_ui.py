"""Optional local browser smoke check. Install playwright and its chromium browser first.
Run from repo root with the API running on 127.0.0.1:8000.
"""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:8000")
    page.wait_for_function("document.querySelector('#status').textContent.includes('Model ready')")
    tabs = page.get_by_role('tab')
    assert tabs.count() == 3
    assert tabs.nth(0).get_attribute('aria-selected') == 'true'
    tabs.nth(0).focus()
    page.keyboard.press('ArrowRight')
    assert tabs.nth(1).get_attribute('aria-selected') == 'true'
    assert page.locator('#evaluation').is_visible()
    assert page.evaluate('document.activeElement.id') == 'tab-evaluation'
    page.keyboard.press('End')
    assert tabs.nth(2).get_attribute('aria-selected') == 'true'
    assert page.locator('#method').is_visible()
    page.keyboard.press('ArrowLeft')
    assert tabs.nth(1).get_attribute('aria-selected') == 'true'
    page.keyboard.press('Home')
    assert tabs.nth(0).get_attribute('aria-selected') == 'true'
    assert page.locator('#experiment').is_visible()
    with page.expect_response("**/api/predict") as response:
        page.locator('#run').click()
    assert response.value.ok
    page.locator('#result .score').first.wait_for()
    Path('docs').mkdir(exist_ok=True)
    page.screenshot(path='docs/desktop.png', full_page=True)
    page.locator('[data-tab=evaluation]').click()
    assert page.locator('#evaluation').is_visible()
    assert page.locator('#comparison tr').count() == 4
    assert '3 of 33 fatal-labelled validation rows' in page.locator('#fatal-audit-summary').inner_text()
    assert page.locator('#fatal-audit-breakdown tr').count() == 2
    assert 'exploratory' in page.locator('#fatal-audit-note').inner_text().lower()
    assert '5-fold StratifiedGroupKFold' in page.locator('#cv-summary').inner_text()
    assert 'fatal rows per fold' in page.locator('#cv-summary').inner_text()
    assert 'final test holdout is excluded' in page.locator('#cv-note').inner_text().lower()
    assert page.locator('#cv-results tr').count() == 4
    assert page.locator('#cv-results tr').first.locator('td').count() == 4
    assert 'brier score' in page.locator('#fatal-calibration-summary').inner_text().lower()
    assert page.locator('#fatal-calibration-rows tr').count() > 0
    assert 'no calibration transform is fitted' in page.locator('#fatal-calibration-note').inner_text().lower()
    assert page.locator('#fatal-threshold-model option').count() == 4
    assert '11/31 test cases' in page.locator('#fatal-recall-ci').inner_text()
    assert 'wilson interval' in page.locator('#fatal-recall-ci').inner_text().lower()
    assert 'fatal recall' in page.locator('#fatal-threshold-scores').inner_text().lower()
    assert 'nonfatal false-alarm rate' in page.locator('#fatal-threshold-scores').inner_text().lower()
    page.locator('#fatal-threshold-slider').evaluate("element => { element.value = '0.25'; element.dispatchEvent(new Event('input', { bubbles: true })); }")
    assert page.locator('#fatal-threshold-value').inner_text() == '0.25'
    assert 'fatal rows flagged' in page.locator('#fatal-threshold-counts').inner_text().lower()
    page.locator('#fatal-threshold-model').select_option('balanced_logistic')
    assert 'fatal rows flagged' in page.locator('#fatal-threshold-counts').inner_text().lower()
    assert 'no cutoff is selected' in page.locator('.threshold-explorer').inner_text().lower()
    with page.expect_download() as download_info:
        page.locator('#export-fatal-cutoffs').click()
    download = download_info.value
    assert download.suggested_filename == 'roadlab-balanced_logistic-fatal-cutoffs.csv'
    download.save_as('/tmp/roadlab-fatal-cutoffs.csv')
    csv_text = Path('/tmp/roadlab-fatal-cutoffs.csv').read_text()
    assert csv_text.startswith('\"threshold\",\"fatal_detected\"')
    assert len(csv_text.splitlines()) == 22
    assert page.locator('#interval .interval-card').count() == 2
    interval_cards = page.locator('#interval .interval-card')
    assert 'stratified row bootstrap' in interval_cards.nth(0).inner_text().lower()
    assert '300 resamples' in interval_cards.nth(0).inner_text()
    assert 'profile-cluster sensitivity' in interval_cards.nth(1).inner_text().lower()
    assert '2,258 test profiles' in interval_cards.nth(1).inner_text()
    assert 'not incident IDs' in page.locator('#interval-note').inner_text()
    page.locator('.threshold-explorer').screenshot(path='docs/evidence.png')
    page.locator('[data-tab=method]').click()
    assert page.locator('#method').is_visible()
    page.locator('[data-tab=experiment]').click()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.locator('[data-tab=evaluation]').click()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='docs/mobile.png', full_page=False)
    assert not errors, errors
    browser.close()
    print(json.dumps({"live_inference": "passed", "tabs": "passed", "mobile_overflow": False, "javascript_errors": errors}))
