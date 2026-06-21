"""
Colab 자동화 — Chrome Profile 복사 + 직접 file input 설정 방식
"""
import time, shutil, tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

NOTEBOOK_PATH  = Path(__file__).parent / "korean_clip_colab.ipynb"
CHROME_PROFILE = Path.home() / "Library/Application Support/Google/Chrome/Profile 8"
CHROME_EXE     = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_IGNORE = shutil.ignore_patterns(
    "Cache", "CachedData", "CachedExtensions", "GPUCache", "ShaderCache",
    "DawnGraphiteCache", "DawnWebGPUCache", "VideoDecodeStats",
    "blob_storage", "Crashpad", "BrowserMetrics*",
)

def log(msg): print(f"[Colab] {msg}", flush=True)


def upload_notebook(page) -> bool:
    """
    파일 → 노트 업로드 후 Colab 자체 업로드 다이얼로그의
    input[type=file] 에 직접 파일 설정
    """
    # 파일 메뉴 열기
    log("파일 메뉴 클릭...")
    page.get_by_role("menubar").get_by_text("파일").click(timeout=10000)
    page.wait_for_timeout(1000)

    # 노트 업로드 클릭 (file chooser 없이)
    log("노트 업로드 항목 클릭...")
    page.get_by_role("menuitem").filter(has_text="업로드").first.click(timeout=5000)
    page.wait_for_timeout(2000)

    # Colab 업로드 다이얼로그의 input[type=file] 찾기
    log("파일 input 탐색...")
    for attempt in range(5):
        try:
            inp = page.locator("input[type=file]").first
            inp.set_input_files(str(NOTEBOOK_PATH))
            log(f"파일 설정 완료: {NOTEBOOK_PATH.name}")
            page.wait_for_timeout(4000)
            return True
        except Exception as e:
            log(f"시도 {attempt+1} 실패: {e}")
            page.wait_for_timeout(1000)

    return False


def set_gpu_runtime(page):
    log("GPU 런타임 설정...")
    try:
        page.get_by_role("menubar").get_by_text("런타임").click(timeout=8000)
        page.wait_for_timeout(600)
        page.get_by_role("menuitem").filter(has_text="런타임 유형 변경").click(timeout=5000)
        page.wait_for_timeout(2000)

        for gpu in ["T4 GPU", "A100 GPU", "T4", "A100"]:
            try:
                btn = page.get_by_text(gpu, exact=True).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    log(f"GPU 선택: {gpu}")
                    break
            except Exception:
                continue

        try:
            page.get_by_role("button", name="저장").click(timeout=5000)
        except Exception:
            page.keyboard.press("Enter")
        page.wait_for_timeout(2000)
        log("런타임 저장 완료")
    except Exception as e:
        log(f"GPU 설정 실패: {e}")


def run_all_cells(page):
    log("전체 셀 실행...")
    try:
        page.get_by_role("menubar").get_by_text("런타임").click(timeout=8000)
        page.wait_for_timeout(600)
        page.get_by_role("menuitem").filter(has_text="모두 실행").click(timeout=5000)
        log("실행 시작!")
        page.wait_for_timeout(3000)
    except Exception as e:
        log(f"메뉴 실행 실패: {e} — 단축키 시도")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        page.keyboard.press("Control+F9")
        page.wait_for_timeout(3000)
        log("단축키로 실행 (Ctrl+F9)")


def allow_drive(page):
    log("Drive 연결 팝업 대기...")
    time.sleep(8)
    for btn_text in ["Google Drive에 연결", "허용", "Allow", "Connect to Google Drive"]:
        try:
            btn = page.get_by_role("button", name=btn_text).first
            if btn.is_visible(timeout=3000):
                btn.click()
                log(f"Drive 허용: {btn_text}")
                page.wait_for_timeout(2000)
                return
        except Exception:
            continue


def monitor(page, hours=4):
    log(f"모니터링 시작 (최대 {hours}시간)")
    start, last_log = time.time(), ""
    while time.time() - start < hours * 3600:
        mins = int(time.time() - start) // 60
        try:
            outputs = page.locator(".output_subarea, .cell_output, pre").all_text_contents()
            text = " ".join(outputs)
            if "===== 완료 =====" in text:
                log(f"학습 완료! ({mins}분)")
                return "completed"
            if "❌ 배포 거부" in text:
                log(f"벤치마크 미통과 ({mins}분)")
                return "rejected"
            for err in page.locator(".error, .ansi-red-fg").all_text_contents():
                if len(err) > 20 and "Warning" not in err and err != last_log:
                    log(f"오류: {err[:150]}")
                    last_log = err
            if mins % 2 == 0 and int(time.time() - start) % 20 < 3:
                recent = [t.strip() for t in outputs if t.strip()]
                if recent:
                    last = recent[-1][-120:]
                    if last != last_log:
                        log(f"[{mins}분] {last}")
                        last_log = last
        except Exception:
            pass
        time.sleep(20)
    log("타임아웃")
    return "timeout"


def run():
    tmp = Path(tempfile.mkdtemp(prefix="exxas_colab_"))
    log("Chrome 프로필 복사 중...")
    shutil.copytree(str(CHROME_PROFILE), str(tmp / "Default"), ignore=_IGNORE)
    log("복사 완료")

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=str(tmp),
            headless=False,
            executable_path=CHROME_EXE if Path(CHROME_EXE).exists() else None,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled",
                  "--window-size=1400,900"],
            slow_mo=300,
            viewport={"width": 1400, "height": 900},
        )
        page = browser.new_page()

        try:
            log("Colab 접속...")
            page.goto("https://colab.research.google.com",
                      timeout=40000, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)

            if "accounts.google.com" in page.url:
                log("로그인 필요 — 브라우저에서 로그인 후 Enter 입력")
                input("로그인 후 Enter...")
                page.wait_for_timeout(3000)

            # 업로드
            ok = upload_notebook(page)
            if not ok:
                log("자동 업로드 실패 — 수동 업로드 후 Enter 입력")
                log(f"파일: {NOTEBOOK_PATH}")
                input("업로드 완료 후 Enter...")

            # 셀 로드 대기
            try:
                page.wait_for_selector(".cell, .codecell", timeout=20000)
                log("노트북 로드 완료")
            except PWTimeout:
                log("셀 로드 타임아웃 — 계속")

            set_gpu_runtime(page)
            run_all_cells(page)
            allow_drive(page)

            status = monitor(page, hours=4)
            log(f"최종 상태: {status}")
            if status == "completed":
                log("Google Drive: 내 드라이브/EXXAS_CLIP/model/ 에 저장됨")

            log("10분 후 종료")
            time.sleep(600)

        except KeyboardInterrupt:
            log("중단")
        except Exception as e:
            log(f"오류: {e}")
            import traceback; traceback.print_exc()
            log("브라우저 열린 상태 유지")
            time.sleep(600)
        finally:
            try: browser.close()
            except Exception: pass

    shutil.rmtree(str(tmp), ignore_errors=True)


if __name__ == "__main__":
    run()
