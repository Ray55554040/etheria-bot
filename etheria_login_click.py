# -*- coding: utf-8 -*-
"""
Etheria Helper — login / skip / avatar 自動化流程
用法：
  python etheria_login_click.py login            只做登入點擊
  python etheria_login_click.py skip             只做跳過
  python etheria_login_click.py avatar           只做選女生頭像
  python etheria_login_click.py dice             只做骰子 + 紅色確認
  python etheria_login_click.py custom           選頭像 → 骰子 → 等待 → 跳過
  python etheria_login_click.py                  依序執行 login → custom

指令後可接 key=value 形式覆寫參數，方便除錯：
  python etheria_login_click.py custom SKIP_INITIAL_DELAY_RANGE=(12,18)
"""

import ast
import io
import os
import random
import subprocess
import sys
import time
import pathlib

import numpy as np
from PIL import Image
import cv2 as cv

# ---------- 固定 ADB ----------
ADB_BIN = r"C:\\Program Files\\BlueStacks_nxt\\HD-Adb.exe"
ADB_SERIAL = "127.0.0.1:5625"

# ---------- 路徑 ----------
BASE = pathlib.Path(__file__).resolve().parent
ASSETS = BASE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

LOGO_TPL   = ASSETS / "login_logo.png"
STRIP_TPL  = ASSETS / "tap_to_start_strip.png"
SKIP_TPL   = ASSETS / "btn_skip.png"
AVATAR_FEMALE_TPL = ASSETS / "avatar_female.png"
DICE_TPL   = ASSETS / "btn_dice.png"
CONFIRM_TPL = ASSETS / "btn_confirm_red.png"

# ---------- 參數 ----------
MATCH_THR_LOGIN = 0.60
ALT_THR_LOGIN   = 0.55
MAX_RETRY_LOGIN = 15
SAFE_MARGIN     = 80
LOGO_Y_FRAC     = 0.50

MATCH_THR_SKIP  = 0.65
MAX_RETRY_SKIP  = 10
WAIT_AFTER_TAP  = 0.40
TAP_JITTER_PX   = 5
SKIP_INITIAL_DELAY_RANGE = (10.0, 20.0)

MATCH_THR_AVATAR = 0.70
MAX_RETRY_AVATAR = 10
AVATAR_AFTER_TAP_WAIT = 0.6
AVATAR_VERIFY_RETRY = 6
AVATAR_VERIFY_INTERVAL = 0.5
FEMALE_AVATAR_TAP_FRAC = (0.75, 0.52)

MATCH_THR_DICE = 0.70
MATCH_THR_CONFIRM = 0.68
MAX_ROLL_RETRY = 6
DICE_WAIT_AFTER_ROLL = 0.45
CONFIRM_WAIT_AFTER_TAP = 0.50
DICE_RETRY_INTERVAL = 0.8
DICE_PERSIST_THRESHOLD = 0.55
DICE_TAP_FRAC = (0.50, 0.64)
CONFIRM_TAP_FRAC = (0.50, 0.80)

_TEMPLATE_CACHE = {}
_SCREEN_SIZE = None

# ---------- ADB ----------
def adb(*args, timeout=20):
    cmd = [ADB_BIN, "-s", ADB_SERIAL] + [str(arg) for arg in args]
    return subprocess.check_output(cmd, timeout=timeout)


def screencap():
    global _SCREEN_SIZE
    data = adb("exec-out", "screencap", "-p", timeout=40)
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    _SCREEN_SIZE = img.size
    return img


def get_screen_size(refresh=False):
    global _SCREEN_SIZE
    if refresh or _SCREEN_SIZE is None:
        img = screencap()
        _SCREEN_SIZE = img.size
    return _SCREEN_SIZE


def clamp_point(x, y, sw=None, sh=None):
    if sw is None or sh is None:
        if _SCREEN_SIZE is None:
            return int(round(x)), int(round(y))
        sw, sh = _SCREEN_SIZE
    return max(0, min(int(round(x)), sw - 1)), max(0, min(int(round(y)), sh - 1))


def tap(x, y):
    xi = int(round(x))
    yi = int(round(y))
    xi += random.randint(-TAP_JITTER_PX, TAP_JITTER_PX)
    yi += random.randint(-TAP_JITTER_PX, TAP_JITTER_PX)
    xi, yi = clamp_point(xi, yi)
    adb("shell", "input", "tap", str(int(xi)), str(int(yi)))


def keyevent(code):
    adb("shell", "input", "keyevent", str(code))


def go_home():
    print("[FAILSAFE] BACK → HOME")
    keyevent(4)
    time.sleep(0.3)
    keyevent(3)


# ---------- 影像工具 ----------
def cv_imread_unicode(path: pathlib.Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_UNCHANGED)


def to_bgr_u8(img):
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
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def load_template_cached(path):
    path_obj = pathlib.Path(path)
    if not path_obj.is_absolute():
        path_obj = (BASE / path_obj).resolve()
    try:
        mtime = path_obj.stat().st_mtime
    except FileNotFoundError:
        _TEMPLATE_CACHE.pop(path_obj, None)
        return None
    cached = _TEMPLATE_CACHE.get(path_obj)
    if cached and cached[0] == mtime:
        return cached[1]
    tpl = cv_imread_unicode(path_obj)
    tpl = to_bgr_u8(tpl)
    _TEMPLATE_CACHE[path_obj] = (mtime, tpl)
    return tpl


def multiscale_match(hay_bgr, tpl_img, thr=0.6, scales=(0.8, 0.9, 1.0, 1.1, 1.2)):
    if tpl_img is None:
        return 0.0, None
    hay = to_bgr_u8(hay_bgr)
    tpl0 = to_bgr_u8(tpl_img)
    if hay is None or tpl0 is None or tpl0.size == 0:
        return 0.0, None

    try:
        thr_value = float(thr)
    except (TypeError, ValueError):
        thr_value = None

    best = (0.0, None)
    for s in scales:
        th = max(5, int(tpl0.shape[0] * s))
        tw = max(5, int(tpl0.shape[1] * s))
        if th < 5 or tw < 5:
            continue
        t = cv.resize(tpl0, (tw, th), interpolation=cv.INTER_AREA if s < 1 else cv.INTER_CUBIC)
        res = cv.matchTemplate(hay, t, cv.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv.minMaxLoc(res)
        if maxv > best[0]:
            cx, cy = maxl[0] + tw // 2, maxl[1] + th // 2
            best = (maxv, (cx, cy))
            if thr_value is not None and maxv >= thr_value:
                break
    return best


def grab_screen():
    img = screencap()
    bgr = cv.cvtColor(np.array(img), cv.COLOR_RGBA2BGR)
    return img, bgr


def pick_wait_seconds(cfg):
    if cfg is None:
        return None
    try:
        if isinstance(cfg, (int, float)):
            return max(0.0, float(cfg))
        if isinstance(cfg, (list, tuple)):
            if not cfg:
                return None
            if len(cfg) == 1:
                return max(0.0, float(cfg[0]))
            lo = float(cfg[0])
            hi = float(cfg[1])
            if hi < lo:
                lo, hi = hi, lo
            return max(0.0, random.uniform(lo, hi))
    except (TypeError, ValueError):
        return None
    return None


# ---------- 登入 ----------
def decide_login_tap():
    img, bgr = grab_screen()
    sw, sh = img.size

    tpl_logo = load_template_cached(LOGO_TPL)
    if tpl_logo is not None:
        score, center = multiscale_match(bgr, tpl_logo, MATCH_THR_LOGIN)
        if score >= MATCH_THR_LOGIN and center:
            return center

    tpl_strip = load_template_cached(STRIP_TPL)
    if tpl_strip is not None:
        score, center = multiscale_match(bgr, tpl_strip, ALT_THR_LOGIN)
        if score >= ALT_THR_LOGIN and center:
            th = tpl_strip.shape[0] if hasattr(tpl_strip, "shape") else int(sh * 0.08)
            top = center[1] - th // 2
            y = min(int(sh * LOGO_Y_FRAC), max(10, top - SAFE_MARGIN))
            return (sw // 2, y)
    return None


def do_login():
    print("=== LOGIN ===")
    for i in range(MAX_RETRY_LOGIN):
        pos = decide_login_tap()
        if pos:
            print(f"[{i}] 點擊登入 {pos}")
            tap(*pos)
            time.sleep(0.5)
            tap(*pos)
            print("[OK] 登入點擊完成")
            return True
        print(f"[{i}] 未偵測登入畫面…")
        time.sleep(0.7)
    print("[WARN] 登入失敗 → 回桌面")
    go_home()
    return False


# ---------- 選擇女生頭像 ----------
def select_female_avatar():
    print("=== AVATAR ===")
    tpl_avatar = load_template_cached(AVATAR_FEMALE_TPL)
    tpl_dice = load_template_cached(DICE_TPL)
    warned_missing_avatar_tpl = False

    for attempt in range(1, MAX_RETRY_AVATAR + 1):
        img, bgr = grab_screen()
        sw, sh = img.size
        tapped = False
        if tpl_avatar is not None:
            score, center = multiscale_match(bgr, tpl_avatar, MATCH_THR_AVATAR)
            print(f"[AVATAR] score={score:.3f}")
            if score >= MATCH_THR_AVATAR and center:
                tap(*center)
                tapped = True
        else:
            if not warned_missing_avatar_tpl:
                print("[AVATAR] 尚無 avatar_female.png 模板，改用預設座標")
                warned_missing_avatar_tpl = True
        if not tapped:
            xr, yr = FEMALE_AVATAR_TAP_FRAC
            x = int(sw * xr)
            y = int(sh * yr)
            print(f"[AVATAR] 使用預設座標({x},{y})")
            tap(x, y)
        time.sleep(AVATAR_AFTER_TAP_WAIT)
        if tpl_dice is None:
            print("[AVATAR] 無骰子模板可驗證，假設已進入下一步")
            return True
        for verify in range(AVATAR_VERIFY_RETRY):
            img2, bgr2 = grab_screen()
            score_dice, _ = multiscale_match(bgr2, tpl_dice, MATCH_THR_DICE)
            print(f"[AVATAR] 驗證骰子 score={score_dice:.3f}")
            if score_dice >= MATCH_THR_DICE:
                print("[AVATAR] 成功進入骰子畫面")
                return True
            time.sleep(AVATAR_VERIFY_INTERVAL)
        print(f"[AVATAR] 第{attempt}次未偵測到骰子畫面 → 重試")
    print("[AVATAR] 多次重試仍失敗 → 回桌面")
    go_home()
    return False


# ---------- 骰子 + 紅色確認 ----------
def roll_dice_and_confirm():
    print("=== DICE ===")
    tpl_dice = load_template_cached(DICE_TPL)
    tpl_confirm = load_template_cached(CONFIRM_TPL)
    warned_missing_dice_tpl = tpl_dice is None
    warned_missing_confirm_tpl = tpl_confirm is None

    for attempt in range(1, MAX_ROLL_RETRY + 1):
        img, bgr = grab_screen()
        sw, sh = img.size
        if tpl_dice is not None:
            score_dice, center_dice = multiscale_match(bgr, tpl_dice, MATCH_THR_DICE)
            print(f"[DICE] 第{attempt}輪 score={score_dice:.3f}")
            if score_dice >= MATCH_THR_DICE and center_dice:
                tap(*center_dice)
            else:
                x = int(sw * DICE_TAP_FRAC[0])
                y = int(sh * DICE_TAP_FRAC[1])
                print(f"[DICE] 使用預設骰子座標({x},{y})")
                tap(x, y)
        else:
            if not warned_missing_dice_tpl:
                print("[DICE] 尚無 btn_dice.png 模板，改用預設座標")
                warned_missing_dice_tpl = True
            x = int(sw * DICE_TAP_FRAC[0])
            y = int(sh * DICE_TAP_FRAC[1])
            tap(x, y)

        time.sleep(DICE_WAIT_AFTER_ROLL)

        img2, bgr2 = grab_screen()
        sw, sh = img2.size
        if tpl_confirm is not None:
            score_confirm, center_confirm = multiscale_match(bgr2, tpl_confirm, MATCH_THR_CONFIRM)
            print(f"[DICE] 確認 score={score_confirm:.3f}")
            if score_confirm >= MATCH_THR_CONFIRM and center_confirm:
                tap(*center_confirm)
            else:
                x = int(sw * CONFIRM_TAP_FRAC[0])
                y = int(sh * CONFIRM_TAP_FRAC[1])
                print(f"[DICE] 使用預設確認座標({x},{y})")
                tap(x, y)
        else:
            if not warned_missing_confirm_tpl:
                print("[DICE] 尚無 btn_confirm_red.png 模板，改用預設座標")
                warned_missing_confirm_tpl = True
            x = int(sw * CONFIRM_TAP_FRAC[0])
            y = int(sh * CONFIRM_TAP_FRAC[1])
            tap(x, y)

        time.sleep(CONFIRM_WAIT_AFTER_TAP)

        if tpl_dice is None and tpl_confirm is None:
            print("[DICE] 無模板可驗證，假設已完成")
            return True

        img3, bgr3 = grab_screen()
        success = True
        if tpl_dice is not None:
            score_after, _ = multiscale_match(bgr3, tpl_dice, DICE_PERSIST_THRESHOLD)
            print(f"[DICE] 驗證骰子殘留 score={score_after:.3f}")
            if score_after >= DICE_PERSIST_THRESHOLD:
                success = False
        if tpl_confirm is not None and success:
            score_confirm_after, _ = multiscale_match(bgr3, tpl_confirm, MATCH_THR_CONFIRM)
            print(f"[DICE] 驗證確認殘留 score={score_confirm_after:.3f}")
            if score_confirm_after >= MATCH_THR_CONFIRM:
                success = False
        if success:
            print("[DICE] 完成骰子與確認")
            return True

        print(f"[DICE] 第{attempt}輪未成功 → 再試一次")
        time.sleep(DICE_RETRY_INTERVAL)

    print("[DICE] 多次重試仍失敗 → 回桌面")
    go_home()
    return False


# ---------- 跳過 ----------
def do_skip(initial_delay_range=None):
    print("=== SKIP ===")
    wait_seconds = pick_wait_seconds(initial_delay_range)
    if wait_seconds is not None and wait_seconds > 0:
        print(f"[SKIP] 等待 {wait_seconds:.2f}s 再開始尋找『跳過』")
        time.sleep(wait_seconds)

    seq = [(0.50, 0.55), (0.93, 0.08), (0.50, 0.70), (0.93, 0.10)]
    warned_missing_tpl = False

    for i in range(MAX_RETRY_SKIP):
        sw, sh = get_screen_size()
        xr, yr = seq[i % len(seq)]
        x = int(sw * xr)
        y = int(sh * yr)
        print(f"[SKIP] 喚醒點擊({x},{y})")
        tap(x, y)
        time.sleep(WAIT_AFTER_TAP)

        img, bgr = grab_screen()
        tpl = load_template_cached(SKIP_TPL)
        if tpl is not None:
            score, center = multiscale_match(bgr, tpl, MATCH_THR_SKIP)
            print(f"[SKIP] score={score:.3f}")
            if score >= MATCH_THR_SKIP and center:
                print(f"[SKIP] 偵測到『跳過』 → 連點兩下 {center}")
                tap(*center)
                time.sleep(0.30)
                tap(*center)
                return True
        else:
            if not warned_missing_tpl:
                print("[SKIP] 尚無 btn_skip.png 模板，僅做喚醒點擊…")
                warned_missing_tpl = True

    print("[WARN] 仍未成功找到『跳過』 → 回桌面")
    go_home()
    return False


# ---------- 綜合流程 ----------
def do_custom_flow():
    if not select_female_avatar():
        return False
    if not roll_dice_and_confirm():
        return False
    return do_skip(initial_delay_range=SKIP_INITIAL_DELAY_RANGE)


# ---------- CLI 覆寫 ----------
def apply_cli_overrides(items):
    global _TEMPLATE_CACHE
    for raw in items:
        if "=" not in raw:
            print(f"[DEBUG] 忽略未識別的覆寫參數：{raw}")
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key not in globals():
            print(f"[WARN] 無此變數可覆寫：{key}")
            continue
        try:
            new_val = ast.literal_eval(value)
        except Exception:
            new_val = value
        current = globals()[key]
        if isinstance(current, pathlib.Path) and not isinstance(new_val, pathlib.Path):
            new_val = pathlib.Path(new_val)
        globals()[key] = new_val
        if key.endswith("_TPL"):
            _TEMPLATE_CACHE.clear()
        print(f"[DEBUG] Override {key} -> {new_val!r}")


VALID_MODES = {"login", "skip", "avatar", "dice", "custom", "all"}


def parse_mode_and_overrides(argv):
    if not argv:
        return "all", []
    first = argv[0]
    if "=" in first:
        return "all", argv
    mode = first.lower()
    if mode not in VALID_MODES:
        print(f"[WARN] 未知模式：{first!r} → 改用 'all'")
        return "all", argv[1:]
    return mode, argv[1:]


# ---------- 主 ----------
def main():
    mode, overrides = parse_mode_and_overrides(sys.argv[1:])
    if overrides:
        apply_cli_overrides(overrides)

    if mode == "login":
        do_login()
        return
    if mode == "skip":
        do_skip()
        return
    if mode == "avatar":
        select_female_avatar()
        return
    if mode == "dice":
        roll_dice_and_confirm()
        return
    if mode == "custom":
        do_custom_flow()
        return

    # 預設 all：登入 → 自訂流程
    if not do_login():
        return
    do_custom_flow()


if __name__ == "__main__":
    main()
