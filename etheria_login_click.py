# -*- coding: utf-8 -*-
"""
Etheria Helper — login / skip 可獨立執行（修正模板型別/通道不一致）
用法：
  python etheria_login_click.py login   只做登入點擊
  python etheria_login_click.py skip    只做跳過
  python etheria_login_click.py         先 login 再 skip
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

# ---------- 主 ----------
def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "all")
    if mode in ("login", "all"):
        ok = do_login()
        if mode == "login":
            return
        if not ok:
            return
    if mode in ("skip", "all"):
        do_skip()

if __name__ == "__main__":
    main()
