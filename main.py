# main.py

"""
Robot Arm – Vision-Guided Pick System
======================================
Alur kerja:
  1. COLOR_SELECT → pilih warna target
  2. IDLE         → tekan [S] untuk mulai
  3. SCAN_E       → sweep sumbu E mencari objek
  4. SCAN_Y       → jika E habis belum nemu, geser Y lalu ulangi
  5. ALIGN        → koreksi X/Y ke center, kecepatan adaptif 3 zona
  6. PICK         → kompensasi offset kamera→gripper, turun, ambil, HOME
  7. DONE         → tekan [S] untuk ulangi

Tombol:
  [1-6]   – pilih warna preset  (saat COLOR_SELECT)
  [C]     – kalibrasi warna live (arahkan ke objek, ENTER konfirmasi)
  [S]     – mulai / reset
  [Q]     – keluar
  Klik kiri di jendela Vision → set ulang CENTER crosshair
"""

import time
import cv2
import numpy as np

import vision_robobio as vis
from vision_robobio import (
    detection, draw_overlay, pixel_to_mm, pixel_error,
    draw_calib_roi, calibrate_from_roi, save_calibrated_profile,
    COLOR_PROFILES, set_color_profile, get_active_profile_name,
    MIN_AREA_PICK, CENTER_TOLERANCE,
)
import kinematics_robobio as kin
from kinematics_robobio import (
    serial_connect, serial_close,
    move, dwell, home, gripper_open, gripper_close,
    adaptive_feed, pos,
    E_MIN, E_MAX, E_STEP,
    Y_HOME, Y_MIN, Y_MAX, Y_STEP,
    X_HOME, X_MIN, X_MAX,
    Z_SEARCH, Z_PICK_HOVER, Z_PICK_DOWN,
    FEED_SEARCH, FEED_PICK,
    CAMERA_OFFSET_X_MM, CAMERA_OFFSET_Y_MM,
)

# ── STATE MACHINE ─────────────────────────────────────────────────
STATE_COLOR_SELECT = "COLOR_SELECT"
STATE_IDLE         = "IDLE"
STATE_SCAN_E       = "SCAN_E"
STATE_SCAN_Y       = "SCAN_Y"
STATE_ALIGN        = "ALIGN"
STATE_PICK         = "PICK"
STATE_DONE         = "DONE"

# ── COLOR SELECTOR ────────────────────────────────────────────────
_COLOR_KEYS = {
    ord("1"): "yellow",
    ord("2"): "red",
    ord("3"): "green",
    ord("4"): "blue",
    ord("5"): "orange",
    ord("6"): "white",
}

_HUD_COL = {
    "yellow": (0,   220, 220),
    "red":    (60,   60, 220),
    "green":  (60,  200,  60),
    "blue":   (220, 100,  60),
    "orange": (30,  160, 230),
    "white":  (220, 220, 220),
    "custom": (180, 100, 255),
}


def run_color_select(cap: cv2.VideoCapture) -> str:
    """
    Loop kamera pra-start untuk memilih warna target.
    Return nama profil yang dikonfirmasi.
    """
    calib_mode = False
    current    = get_active_profile_name()
    presets    = list(_COLOR_KEYS.values())

    print("\n[COLOR SELECT] Pilih warna objek:")
    for i, name in enumerate(presets, 1):
        print(f"  [{i}] {name}")
    print("  [C] Kalibrasi live dari kamera")
    print("  [ENTER] Konfirmasi\n")

    cv2.namedWindow("Color Select")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Preview mask di pojok kiri atas
        _, _, _, masked, *_ = detection(frame.copy())
        thumb = cv2.resize(masked, (frame.shape[1] // 3, frame.shape[0] // 3))
        frame[0:thumb.shape[0], 0:thumb.shape[1]] = thumb

        # Panel bawah
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 140), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        y0 = h - 132
        cv2.putText(frame, "PILIH WARNA OBJEK", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        for i, name in enumerate(presets):
            col    = _HUD_COL.get(name, (200, 200, 200))
            marker = ">" if name == current else " "
            cv2.putText(frame, f"{marker}[{i+1}] {name}",
                        (10 + (i % 3) * 190, y0 + 22 + (i // 3) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        cv2.putText(frame, "[C] Kalibrasi  [ENTER] Mulai",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
        cv2.putText(frame, f"Aktif: {current.upper()}",
                    (w - 200, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    _HUD_COL.get(current, (200, 200, 200)), 2)

        if calib_mode:
            frame = draw_calib_roi(frame)
            cv2.putText(frame, "Arahkan ke objek. ENTER=simpan  ESC=batal",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

        cv2.imshow("Color Select", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in _COLOR_KEYS and not calib_mode:
            current = _COLOR_KEYS[key]
            set_color_profile(current)
        elif key in (ord("c"), ord("C")):
            calib_mode = not calib_mode
            print("[CALIB] Mode kalibrasi", "aktif." if calib_mode else "dibatalkan.")
        elif key == 13:     # ENTER
            if calib_mode:
                low, high  = calibrate_from_roi(frame)
                save_calibrated_profile(low, high, "custom")
                current    = "custom"
                calib_mode = False
                print(f"[CALIB] Profil 'custom' disimpan. Low={low} High={high}")
            else:
                print(f"[COLOR SELECT] Konfirmasi: {current.upper()}")
                break
        elif key == 27 and calib_mode:     # ESC
            calib_mode = False

    cv2.destroyWindow("Color Select")
    return current


# ── HUD ───────────────────────────────────────────────────────────
_STATE_COLOR = {
    STATE_IDLE:         (180, 180, 180),
    STATE_SCAN_E:       (255, 200,   0),
    STATE_SCAN_Y:       (255, 150,   0),
    STATE_ALIGN:        (  0, 255, 150),
    STATE_PICK:         (  0, 100, 255),
    STATE_DONE:         (  0, 255,   0),
    STATE_COLOR_SELECT: (200, 100, 255),
}


def draw_hud(frame, state, cx, cy, dx=0.0, dy=0.0,
             feed_label="", color_name=""):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    sc    = _STATE_COLOR.get(state, (255, 255, 255))
    label = f"STATE: {state}"
    if feed_label:
        label += f"  [{feed_label}]"
    if color_name:
        label += f"  | {color_name.upper()}"
    cv2.putText(frame, label, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, sc, 2)

    cv2.putText(frame,
                f"X:{pos['X']:.1f}  Y:{pos['Y']:.1f}  Z:{pos['Z']:.1f}  E:{pos['E']:.1f}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    if cx != -1:
        cv2.putText(frame,
                    f"obj ({cx},{cy})  err dx={dx:.1f}mm dy={dy:.1f}mm",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)
    return frame


# ── MOUSE CALLBACK ─────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        vis.CENTER_X_MARK = x
        vis.CENTER_Y_MARK = y
        print(f"[CALIB] Center crosshair → ({x}, {y})")


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    serial_connect()
    send_init()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    selected_color = run_color_select(cap)
    set_color_profile(selected_color)

    cv2.namedWindow("Vision")
    cv2.setMouseCallback("Vision", on_mouse)

    state      = STATE_IDLE
    e_current  = E_MIN
    y_current  = Y_HOME
    feed_label = ""

    print("[INFO] Tekan [S] untuk mulai, [Q] untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Kamera tidak terbaca.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[EXIT] Keluar …")
            break
        elif key == ord("s") and state in (STATE_IDLE, STATE_DONE):
            print("[START] Mulai scan …")
            e_current = E_MIN
            y_current = Y_HOME
            move(X=X_HOME, Y=Y_HOME, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)
            time.sleep(0.5)
            state = STATE_SCAN_E

        # ── Deteksi ───────────────────────────────────────────────
        cx, cy, cw, _, x1, y1, x2, y2 = detection(frame)
        dx, dy = (0.0, 0.0)
        ex, ey = (0, 0)
        if cx != -1:
            dx, dy = pixel_to_mm(cx, cy)
            ex, ey = pixel_error(cx, cy)

        object_detected   = cx != -1
        object_centered   = object_detected and abs(ex) <= CENTER_TOLERANCE \
                                             and abs(ey) <= CENTER_TOLERANCE
        object_big_enough = object_detected and cw != -1 and cw >= MIN_AREA_PICK

        # ── STATE MACHINE ─────────────────────────────────────────
        if state == STATE_IDLE:
            pass

        elif state == STATE_SCAN_E:
            if object_detected:
                print(f"[SCAN_E] Objek di ({cx},{cy}). → ALIGN")
                state = STATE_ALIGN
            elif e_current < E_MAX:
                e_current = min(e_current + E_STEP, E_MAX)
                print(f"[SCAN_E] E → {e_current}")
                move(E=e_current, F=FEED_SEARCH)
                time.sleep(0.3)
            else:
                print("[SCAN_E] Sweep selesai, belum nemu. → SCAN_Y")
                e_current = E_MIN
                move(E=e_current, F=FEED_SEARCH)
                state = STATE_SCAN_Y

        elif state == STATE_SCAN_Y:
            y_next = y_current + Y_STEP
            if y_next > Y_MAX:
                print("[SCAN_Y] Area habis. Kembali HOME.")
                home()
                state = STATE_IDLE
            else:
                y_current = y_next
                print(f"[SCAN_Y] Y → {y_current:.1f}. Ulangi sweep E.")
                move(Y=y_current, F=FEED_SEARCH)
                time.sleep(0.3)
                state = STATE_SCAN_E

        elif state == STATE_ALIGN:
            if not object_detected:
                print("[ALIGN] Objek hilang. → SCAN_E")
                feed_label = ""
                state = STATE_SCAN_E
            elif object_centered:
                print("[ALIGN] Objek di tengah. → PICK")
                feed_label = ""
                state = STATE_PICK
            else:
                feed, zone = adaptive_feed(ex, ey)
                feed_label = zone
                new_x = float(np.clip(pos["X"] + dx, X_MIN, X_MAX))
                new_y = float(np.clip(pos["Y"] + dy, Y_MIN, Y_MAX))
                print(f"[ALIGN-{zone}] X:{new_x:.2f} Y:{new_y:.2f}  "
                      f"(px=({ex},{ey}) mm=({dx:.2f},{dy:.2f}))")
                move(X=new_x, Y=new_y, F=feed)
                time.sleep(0.2)

        elif state == STATE_PICK:
            # Kompensasi offset kamera → gripper
            if CAMERA_OFFSET_X_MM != 0.0 or CAMERA_OFFSET_Y_MM != 0.0:
                pick_x = float(np.clip(pos["X"] + CAMERA_OFFSET_X_MM, X_MIN, X_MAX))
                pick_y = float(np.clip(pos["Y"] + CAMERA_OFFSET_Y_MM, Y_MIN, Y_MAX))
                print(f"[PICK] Offset kompensasi → X:{pick_x:.2f} Y:{pick_y:.2f}")
                move(X=pick_x, Y=pick_y, F=FEED_PICK)
                dwell(0.3)

            print("[PICK] Hover …")
            move(Z=Z_PICK_HOVER, F=FEED_PICK)
            dwell(0.3)

            print("[PICK] Turun …")
            move(Z=Z_PICK_DOWN, F=FEED_PICK)
            dwell(0.5)

            print("[PICK] Gripper CLOSE …")
            gripper_close()
            dwell(0.5)

            print("[PICK] Naik …")
            move(Z=Z_SEARCH, F=FEED_PICK)
            dwell(0.3)

            print("[PICK] HOME …")
            home()
            gripper_open()
            dwell(0.5)
            gripper_close()

            state = STATE_DONE
            print("[DONE] Selesai. Tekan [S] untuk ulangi.")

        elif state == STATE_DONE:
            pass

        # ── Render ────────────────────────────────────────────────
        if cx != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = draw_overlay(frame, cx, cy)
        frame = draw_hud(frame, state, cx, cy, dx, dy,
                         feed_label, get_active_profile_name())
        cv2.imshow("Vision", frame)

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    serial_close()
    print("[INFO] Program selesai.")


def send_init():
    from kinematics_robobio import send
    send("G90")
    home()


if __name__ == "__main__":
    main()