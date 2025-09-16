# -*- coding: utf-8 -*-
"""
Etheria Helper — login / skip 可獨立執行（修正模板型別/通道不一致）
用法：
  python etheria_login_click.py login   只做登入點擊
  python etheria_login_click.py skip    只做跳過
  python etheria_login_click.py walk    走到石柱底下
  python etheria_login_click.py         先 login 再 skip
  python etheria_login_click.py login skip walk  依序執行多段流程
"""

import os, io, time, random, subprocess, sys, pathlib
import numpy as np
from PIL import Image
import cv2 as cv

# ---------- 固定 ADB ----------
ADB_BIN = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
ADB_SERIAL = "127.0.0.1:5625"

# ---------- 路徑 ----------
BASE = pathlib.Path(__file__).resolve().parent
ASSETS = BASE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

LOGO_TPL  = ASSETS / "login_logo.png"
STRIP_TPL = ASSETS / "tap_to_start_strip.png"
SKIP_TPL  = ASSETS / "btn_skip.png"

# ---------- 參數 ----------
MATCH_THR_LOGIN = 0.60
ALT_THR_LOGIN   = 0.55
MAX_RETRY_LOGIN = 15
SAFE_MARGIN     = 80
LOGO_Y_FRAC     = 0.50

MATCH_THR_SKIP  = 0.65   # 簡化回成功版：門檻偏寬鬆
MAX_RETRY_SKIP  = 10
WAIT_AFTER_TAP  = 0.40
TAP_JITTER_PX   = 5

# ---------- 走到石柱 ----------
PILLAR_TAP_SEQUENCE = [
    {
        "name": "center_align",
        "target": (0.50, 0.70),
        "taps": 2,
        "timeout": 9.0,
        "min_wait": 1.1,
    },
    {
        "name": "left_pillar",
        "target": (0.37, 0.44),
        "taps": 2,
        "timeout": 12.0,
    },
    {
        "name": "between_pillars",
        "target": (0.50, 0.42),
        "taps": 1,
        "timeout": 8.0,
        "min_wait": 0.9,
    },
    {
        "name": "right_pillar",
        "target": (0.63, 0.44),
        "taps": 2,
        "timeout": 12.0,
    },
]

PILLAR_TAP_GAP           = 0.35
STABLE_CHECK_INTERVAL    = 0.45
STABLE_MIN_WAIT          = 0.9
STABLE_TIMEOUT           = 10.0
STABLE_FRAMES_REQUIRED   = 3
STABLE_DIFF_THRESHOLD    = 4.0

# ---------- ADB ----------
def adb(*args, timeout=20):
    cmd = [ADB_BIN, "-s", ADB_SERIAL] + list(args)
    return subprocess.check_output(cmd, timeout=timeout)

def screencap():
    data = adb("exec-out", "screencap", "-p", timeout=40)
    return Image.open(io.BytesIO(data))

def tap(x, y):
    # 輕微隨機抖動，避免防機器
    x += random.randint(-TAP_JITTER_PX, TAP_JITTER_PX)
    y += random.randint(-TAP_JITTER_PX, TAP_JITTER_PX)
    adb("shell", "input", "tap", str(int(x)), str(int(y)))

def keyevent(code): adb("shell", "input", "keyevent", str(code))

def go_home():
    print("[FAILSAFE] BACK → HOME")
    keyevent(4); time.sleep(0.3); keyevent(3)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def multi_tap(x, y, times=1, gap=0.30):
    times = max(1, int(times))
    for i in range(times):
        tap(x, y)
        if i + 1 < times:
            time.sleep(gap)


def wait_until_stable(
    timeout=STABLE_TIMEOUT,
    min_wait=STABLE_MIN_WAIT,
    check_interval=STABLE_CHECK_INTERVAL,
    diff_threshold=STABLE_DIFF_THRESHOLD,
    stable_frames=STABLE_FRAMES_REQUIRED,
):
    """偵測畫面平均亮度差是否持續低於門檻以判斷角色停止。"""
    start = time.monotonic()
    if min_wait > 0:
        time.sleep(min_wait)

    prev = None
    stable = 0

    while time.monotonic() - start < timeout:
        img = screencap()
        gray = cv.cvtColor(np.array(img), cv.COLOR_RGBA2GRAY)

        if prev is not None:
            diff = cv.absdiff(gray, prev)
            score = float(np.mean(diff))
            print(f"[STABLE] diff={score:.2f} (threshold={diff_threshold})")
            if score < diff_threshold:
                stable += 1
                if stable >= stable_frames:
                    return True
            else:
                stable = 0

        prev = gray
        time.sleep(check_interval)

    return False

# ---------- 影像工具（統一通道/型別） ----------
def cv_imread_unicode(path: pathlib.Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_UNCHANGED)

def to_bgr_u8(img):
    """把灰階/BGRA 轉成 BGR，並確保 uint8。"""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return img
    # 其他奇怪通道數，轉成灰後再轉 BGR
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)

def multiscale_match(hay_bgr, tpl_img, thr=0.6, scales=(0.8, 0.9, 1.0, 1.1, 1.2)):
    """
    將截圖與模板統一為 BGR uint8，做多尺度 matchTemplate。
    回傳 (score, center(x,y))
    """
    if tpl_img is None:
        return 0.0, None
    hay = to_bgr_u8(hay_bgr)
    tpl0 = to_bgr_u8(tpl_img)
    if hay is None or tpl0 is None or tpl0.size == 0:
        return 0.0, None

    best = (0.0, None)
    for s in scales:
        th, tw = max(5, int(tpl0.shape[0]*s)), max(5, int(tpl0.shape[1]*s))
        if th < 5 or tw < 5:
            continue
        t = cv.resize(tpl0, (tw, th), interpolation=cv.INTER_AREA if s < 1 else cv.INTER_CUBIC)
        # 兩者型別/通道一致（皆 BGR uint8）
        res = cv.matchTemplate(hay, t, cv.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv.minMaxLoc(res)
        if maxv > best[0]:
            cx, cy = maxl[0] + tw // 2, maxl[1] + th // 2
            best = (maxv, (cx, cy))
    return best

# ---------- 登入 ----------
def decide_login_tap():
    img = screencap()
    sw, sh = img.size
    bgr = cv.cvtColor(np.array(img), cv.COLOR_RGBA2BGR)

    if LOGO_TPL.exists():
        tpl = cv_imread_unicode(LOGO_TPL)
        s, c = multiscale_match(bgr, tpl, MATCH_THR_LOGIN)
        if s >= MATCH_THR_LOGIN and c:
            return c

    if STRIP_TPL.exists():
        tpl = cv_imread_unicode(STRIP_TPL)
        s, c = multiscale_match(bgr, tpl, ALT_THR_LOGIN)
        if s >= ALT_THR_LOGIN and c:
            # 用伺服器條上緣推估 Logo 點擊高度，避免點到底部框
            tpl_img = to_bgr_u8(cv_imread_unicode(STRIP_TPL))
            th = tpl_img.shape[0] if tpl_img is not None else int(sh * 0.08)
            top = c[1] - th // 2
            y   = min(int(sh * LOGO_Y_FRAC), max(10, top - SAFE_MARGIN))
            return (sw // 2, y)
    return None

def do_login():
    print("=== LOGIN ===")
    for i in range(MAX_RETRY_LOGIN):
        pos = decide_login_tap()
        if pos:
            print(f"[{i}] 點擊登入 {pos}")
            tap(*pos); time.sleep(0.5); tap(*pos)
            print("[OK] 登入點擊完成")
            return True
        else:
            print(f"[{i}] 未偵測登入畫面…")
            time.sleep(0.7)
    print("[WARN] 登入失敗 → 回桌面")
    go_home()
    return False

# ---------- 跳過（簡化成功版 + 通道修正） ----------
def do_skip():
    print("=== SKIP ===")
    # 若沒有模板，直接走喚醒點擊也能觸發「跳過」浮現
    tpl = cv_imread_unicode(SKIP_TPL) if SKIP_TPL.exists() else None

    # 喚醒序列：中央 → 右上 → 中下 → 右上 …
    seq = [(0.50, 0.55), (0.93, 0.08), (0.50, 0.70), (0.93, 0.10)]

    for i in range(MAX_RETRY_SKIP):
        # 1) 先喚醒點一下
        img0 = screencap()
        sw, sh = img0.size
        xr, yr = seq[i % len(seq)]
        x, y = int(sw * xr), int(sh * yr)
        print(f"[SKIP] 喚醒點擊({x},{y})")
        tap(x, y)
        time.sleep(WAIT_AFTER_TAP)

        # 2) 截圖 + 模板比對
        img = screencap()
        bgr = cv.cvtColor(np.array(img), cv.COLOR_RGBA2BGR)

        if tpl is not None:
            score, center = multiscale_match(bgr, tpl, MATCH_THR_SKIP)
            print(f"[SKIP] score={score:.3f}")
            if score >= MATCH_THR_SKIP and center:
                print(f"[SKIP] 偵測到『跳過』 → 連點兩下 {center}")
                tap(*center); time.sleep(0.30); tap(*center)
                return True
        else:
            print("[SKIP] 尚無 btn_skip.png 模板，僅做喚醒點擊…")

    print("[WARN] 10 輪仍未成功 → 回桌面")
    go_home()
    return False


def walk_to_pillars():
    print("=== WALK TO PILLARS ===")
    img = screencap()
    sw, sh = img.size
    steps = len(PILLAR_TAP_SEQUENCE)
    all_ok = True

    for idx, step in enumerate(PILLAR_TAP_SEQUENCE, 1):
        name = step.get("name", f"step{idx}")
        xf, yf = step.get("target", (0.5, 0.5))
        taps = step.get("taps", 1)
        timeout = step.get("timeout", STABLE_TIMEOUT)
        min_wait = step.get("min_wait", STABLE_MIN_WAIT)
        check_interval = step.get("check_interval", STABLE_CHECK_INTERVAL)
        diff_threshold = step.get("diff_threshold", STABLE_DIFF_THRESHOLD)
        stable_frames = step.get("stable_frames", STABLE_FRAMES_REQUIRED)

        x = clamp(int(round(sw * xf)), 0, sw - 1)
        y = clamp(int(round(sh * yf)), 0, sh - 1)

        print(
            f"[PILLAR] Step {idx}/{steps} '{name}': target=({xf:.3f},{yf:.3f}) → tap ({x},{y}) x{taps}"
        )
        multi_tap(x, y, times=taps, gap=PILLAR_TAP_GAP)

        ok = wait_until_stable(
            timeout=timeout,
            min_wait=min_wait,
            check_interval=check_interval,
            diff_threshold=diff_threshold,
            stable_frames=stable_frames,
        )

        if ok:
            print(f"[PILLAR] {name}: 角色已停止")
        else:
            print(f"[WARN] {name}: 等待靜止逾時（繼續下一步）")
            all_ok = False

    if all_ok:
        print("[OK] 兩側石柱走訪完成")
    else:
        print("[WARN] 石柱流程部分未確認，可視情況微調參數")

    return all_ok

# ---------- 主 ----------
def main():
    raw = [arg.lower() for arg in sys.argv[1:]]
    if not raw:
        raw = ["login", "skip"]

    modes = []
    for item in raw:
        if item == "all":
            modes.extend(["login", "skip"])
        else:
            modes.append(item)

    for mode in modes:
        if mode == "login":
            ok = do_login()
            if not ok:
                return
        elif mode == "skip":
            do_skip()
        elif mode == "walk":
            walk_to_pillars()
        else:
            print(f"[WARN] 未知模式：{mode}")

if __name__ == "__main__":
    main()
