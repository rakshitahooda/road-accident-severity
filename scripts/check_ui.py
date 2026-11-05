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
    with page.expect_response("**/api/predict") as response:
        page.locator('#run').click()
    assert response.value.ok
    page.locator('#result .score').first.wait_for()
    Path('docs').mkdir(exist_ok=True)
    page.screenshot(path='docs/desktop.png', full_page=True)
    page.locator('[data-tab=evaluation]').click()
    assert page.locator('#evaluation').is_visible()
    assert page.locator('#comparison tr').count() == 4
    page.screenshot(path='docs/evidence.png', full_page=True)
    page.locator('[data-tab=method]').click()
    assert page.locator('#method').is_visible()
    page.locator('[data-tab=experiment]').click()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='docs/mobile.png', full_page=True)
    assert not errors, errors
    browser.close()
    print(json.dumps({"live_inference": "passed", "tabs": "passed", "mobile_overflow": False, "javascript_errors": errors}))
