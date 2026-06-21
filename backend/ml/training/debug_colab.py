"""
Colab 페이지 구조 디버깅 — 스크린샷 + HTML 덤프
"""
import time, shutil, tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROME_PROFILE = Path.home() / "Library/Application Support/Google/Chrome/Profile 8"
SCREENSHOT_PATH = Path(__file__).parent / "colab_screenshot.png"

_IGNORE = shutil.ignore_patterns(
    "Cache", "CachedData", "CachedExtensions", "GPUCache",
    "ShaderCache", "DawnGraphiteCache", "DawnWebGPUCache",
    "VideoDecodeStats", "blob_storage", "Crashpad", "BrowserMetrics*",
)


def run():
    tmp = Path(tempfile.mkdtemp(prefix="exxas_debug_"))
    dest = tmp / "Default"
    print(f"프로필 복사 중...")
    shutil.copytree(str(CHROME_PROFILE), str(dest), ignore=_IGNORE)
    print(f"완료")

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(tmp),
            headless=False,
            executable_path=chrome_path if Path(chrome_path).exists() else None,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled",
                  "--window-size=1400,900"],
            slow_mo=200,
            viewport={"width": 1400, "height": 900},
        )
        page = browser.new_page()

        print("Colab 접속...")
        page.goto("https://colab.research.google.com", timeout=40000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)

        print(f"현재 URL: {page.url}")

        if "accounts.google.com" in page.url:
            print("로그인 필요! 브라우저에서 로그인 후 Enter 입력")
            input("로그인 후 Enter...")
            page.wait_for_timeout(3000)
            print(f"URL 후: {page.url}")

        # 스크린샷
        page.screenshot(path=str(SCREENSHOT_PATH))
        print(f"스크린샷 저장: {SCREENSHOT_PATH}")

        # 페이지 텍스트 추출
        text = page.locator("body").inner_text()
        print("\n=== 페이지 텍스트 (처음 2000자) ===")
        print(text[:2000])

        # 버튼/탭 목록
        print("\n=== 버튼 목록 ===")
        btns = page.locator("button, [role=tab], [role=button]").all()
        for b in btns[:30]:
            try:
                print(f"  '{b.inner_text()[:50]}' visible={b.is_visible()}")
            except Exception:
                pass

        print("\n=== 링크 목록 ===")
        links = page.locator("a").all()
        for l in links[:20]:
            try:
                txt = l.inner_text().strip()
                if txt:
                    print(f"  '{txt[:50]}'")
            except Exception:
                pass

        print("\n30초 후 종료...")
        time.sleep(30)
        browser.close()

    shutil.rmtree(str(tmp), ignore_errors=True)


if __name__ == "__main__":
    run()
