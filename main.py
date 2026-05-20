"""
Robot Arm – Vision-Guided Pick System
======================================
Alur kerja:
  1. IDLE      → tekan [S] untuk mulai
  2. SCAN_E    → sweep sumbu E (E_MIN → E_MAX, step E_STEP), kamera cari objek
  3. SCAN_Y    → jika E habis tapi belum nemu, geser Y lalu ulangi sweep E
  4. ALIGN     → objek terdeteksi → koreksi X & Y sampai centroid masuk toleransi
  5. PICK      → turun, ambil, naik, HOME
  6. DONE      → selesai, tekan [S] untuk reset ke IDLE

Tombol:
  [S]  – mulai / reset
  [Q]  – keluar program
  Klik kiri mouse di jendela Vision → set ulang CENTER_X/Y_MARK
"""

import time
import serial
import cv2
import numpy as np
from vision_robobio import (
    detection, draw_overlay, pixel_to_mm,
    CENTER_X_MARK, CENTER_Y_MARK, MIN_AREA_PICK, CENTER_TOLERANCE
)

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI SERIAL
# ═══════════════════════════════════════════════════════════════════
PORT    = 'COM4'
BAUD    = 115200
TIMEOUT = None          # blocking readline – robot selalu balas "ok"

# ═══════════════════════════════════════════════════════════════════
#  PARAMETER WORKSPACE ROBOT
# ═══════════════════════════════════════════════════════════════════
# Sumbu E (rel)
E_MIN   =   0
E_MAX   = 200
E_STEP  =  20

# Sumbu Y
Y_HOME  = 216.90
Y_MIN   =  80.0
Y_MAX   = 300.0
Y_STEP  =  20.0         # langkah geser Y saat E-sweep habis

# Sumbu X
X_HOME  =   0.0
X_MIN   = -100.0
X_MAX   =  100.0

# Sumbu Z
Z_SEARCH    =   0
Z_PICK_HOVER = -83
Z_PICK_DOWN  = -96

# Feed-rate (mm/s)
FEED_SEARCH =  30
FEED_ALIGN  =  10
FEED_PICK   =  15

# ═══════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ═══════════════════════════════════════════════════════════════════
STATE_IDLE   = "IDLE"
STATE_SCAN_E = "SCAN_E"
STATE_SCAN_Y = "SCAN_Y"
STATE_ALIGN  = "ALIGN"
STATE_PICK   = "PICK"
STATE_DONE   = "DONE"

# ═══════════════════════════════════════════════════════════════════
#  POSISI ROBOT (tracking software)
# ═══════════════════════════════════════════════════════════════════
pos = {"X": X_HOME, "Y": Y_HOME, "Z": Z_SEARCH, "E": E_MIN}

# ═══════════════════════════════════════════════════════════════════
#  SERIAL HELPERS
# ═══════════════════════════════════════════════════════════════════
ser = None

def serial_connect():
    global ser
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    time.sleep(1.5)          # beri waktu Arduino reset
    print(f"[SERIAL] Terhubung ke {PORT} @ {BAUD}")

def send(cmd: str):
    """Kirim G/M-code dan tunggu balasan 'ok'."""
    full = cmd.strip() + '\r'
    ser.write(full.encode('utf-8'))
    while True:
        resp = ser.readline()
        if b"ok" in resp:
            break

# ═══════════════════════════════════════════════════════════════════
#  MOVE HELPERS
# ═══════════════════════════════════════════════════════════════════
def move(X=None, Y=None, Z=None, E=None, F=FEED_SEARCH):
    parts = []
    if X is not None: parts.append(f"X{X:.3f}");  pos["X"] = X
    if Y is not None: parts.append(f"Y{Y:.3f}");  pos["Y"] = Y
    if Z is not None: parts.append(f"Z{Z:.3f}");  pos["Z"] = Z
    if E is not None: parts.append(f"E{E:.3f}");  pos["E"] = E
    parts.append(f"F{F}")
    send("G0 " + " ".join(parts))

def dwell(seconds: float):
    """G4 – tahan posisi selama `seconds` detik."""
    send(f"G4 S{seconds}")

def home():
    """G28 – robot kembali ke posisi HOME."""
    send("G28")
    pos.update({"X": X_HOME, "Y": Y_HOME, "Z": Z_SEARCH, "E": E_MIN})

def gripper_open():
    send("G160")            # VACUM ON  (sesuai command list)

def gripper_close():
    send("G130")            # VACUM OFF

# ═══════════════════════════════════════════════════════════════════
#  MOUSE CALLBACK – set ulang crosshair tengah
# ═══════════════════════════════════════════════════════════════════
# def on_mouse(event, x, y, flags, param):
#     global CENTER_X_MARK, CENTER_Y_MARK          # type: ignore[name-defined]
#     if event == cv2.EVENT_LBUTTONDOWN:
#         import vision_robobio as v
#         v.CENTER_X_MARK = x
#         v.CENTER_Y_MARK = y
#         print(f"[CALIB] Crosshair diset ke ({x}, {y})")

# ═══════════════════════════════════════════════════════════════════
#  UTILITAS HUD
# ═══════════════════════════════════════════════════════════════════
def draw_hud(frame, state, cx, cy, dx=0.0, dy=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    STATE_COLOR = {
        STATE_IDLE:   (180, 180, 180),
        STATE_SCAN_E: (255, 200,   0),
        STATE_SCAN_Y: (255, 150,   0),
        STATE_ALIGN:  (0,   255, 150),
        STATE_PICK:   (0,   100, 255),
        STATE_DONE:   (0,   255,   0),
    }
    color = STATE_COLOR.get(state, (255, 255, 255))
    cv2.putText(frame, f"STATE: {state}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame,
                f"POS  X:{pos['X']:.1f}  Y:{pos['Y']:.1f}  Z:{pos['Z']:.1f}  E:{pos['E']:.1f}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    if cx != -1:
        cv2.putText(frame, f"obj ({cx},{cy})  err dx={dx:.1f}mm dy={dy:.1f}mm",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1)
    return frame

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    serial_connect()

    # — Absolute mode, homing awal —
    send("G90")
    home()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Vision")
    # cv2.setMouseCallback("Vision", on_mouse)

    state      = STATE_IDLE
    e_current  = E_MIN          # posisi E saat sweep
    y_current  = Y_HOME         # posisi Y saat grid search

    print("[INFO] Tekan [S] untuk mulai, [Q] untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Kamera tidak terbaca.")
            break

        # ── Deteksi objek ──────────────────────────────────────────
        cx, cy, contour_w, frame_masked = detection(frame)
        dx, dy = (0.0, 0.0)
        if cx != -1:
            dx, dy = pixel_to_mm(cx, cy)

        object_detected = (cx != -1)
        object_centered = (
            object_detected
            and abs(cx - CENTER_X_MARK) <= CENTER_TOLERANCE
            and abs(cy - CENTER_Y_MARK) <= CENTER_TOLERANCE
        )
        object_big_enough = (
            object_detected and contour_w != -1 and contour_w >= MIN_AREA_PICK
        )

        # ── STATE MACHINE ──────────────────────────────────────────
        if state == STATE_IDLE:
            pass    # tunggu keypress [S]

        # ── SCAN_E: sweep E dari posisi saat ini hingga E_MAX ──────
        elif state == STATE_SCAN_E:
            if object_detected:
                print(f"[SCAN_E] Objek terdeteksi di ({cx},{cy}). Beralih ke ALIGN.")
                state = STATE_ALIGN

            elif e_current < E_MAX:
                e_current = min(e_current + E_STEP, E_MAX)
                print(f"[SCAN_E] E → {e_current}")
                move(E=e_current, F=FEED_SEARCH)
                time.sleep(0.3)     # beri kamera waktu stabilisasi

            else:
                # E habis, belum nemu → pindah ke SCAN_Y
                print("[SCAN_E] Sweep E selesai, belum nemu. Beralih ke SCAN_Y.")
                e_current = E_MIN
                move(E=e_current, F=FEED_SEARCH)
                state = STATE_SCAN_Y

        # ── SCAN_Y: geser Y satu langkah, lalu ulang sweep E ───────
        elif state == STATE_SCAN_Y:
            y_next = y_current + Y_STEP
            if y_next > Y_MAX:
                # Sudah habis seluruh area, balik ke HOME
                print("[SCAN_Y] Seluruh area sudah di-scan. Objek tidak ditemukan. Kembali HOME.")
                home()
                state = STATE_IDLE
            else:
                y_current = y_next
                print(f"[SCAN_Y] Y → {y_current:.1f}. Mulai ulang sweep E.")
                move(Y=y_current, F=FEED_SEARCH)
                time.sleep(0.3)
                state = STATE_SCAN_E

        # ── ALIGN: koreksi X & Y sampai objek di tengah ────────────
        elif state == STATE_ALIGN:
            if not object_detected:
                # Objek hilang, balik scan
                print("[ALIGN] Objek hilang. Kembali ke SCAN_E.")
                state = STATE_SCAN_E

            elif object_centered:
                print("[ALIGN] Objek sudah di tengah. Beralih ke PICK.")
                state = STATE_PICK

            else:
                # Hitung posisi baru X & Y robot
                new_x = float(np.clip(pos["X"] + dx, X_MIN, X_MAX))
                new_y = float(np.clip(pos["Y"] + dy, Y_MIN, Y_MAX))
                print(f"[ALIGN] Koreksi → X:{new_x:.2f} Y:{new_y:.2f}  (dx={dx:.2f} dy={dy:.2f})")
                move(X=new_x, Y=new_y, F=FEED_ALIGN)
                time.sleep(0.25)

        # ── PICK: turun, hisap, naik ────────────────────────────────
        elif state == STATE_PICK:
            print("[PICK] Menuju hover …")
            move(Z=Z_PICK_HOVER, F=FEED_PICK)
            dwell(0.3)

            print("[PICK] Turun pick …")
            move(Z=Z_PICK_DOWN, F=FEED_PICK)
            dwell(0.5)

            print("[PICK] Vacuum ON …")
            gripper_open()
            dwell(0.5)

            print("[PICK] Naik …")
            move(Z=Z_SEARCH, F=FEED_PICK)
            dwell(0.3)

            print("[PICK] Kembali HOME …")
            home()
            gripper_close()

            state = STATE_DONE
            print("[DONE] Pick selesai. Tekan [S] untuk ulangi.")

        elif state == STATE_DONE:
            pass    # tunggu keypress [S] untuk reset

        # ── RENDER ─────────────────────────────────────────────────
        frame = draw_overlay(frame, cx, cy)
        frame = draw_hud(frame, state, cx, cy, dx, dy)

        cv2.imshow("Vision", frame)
        cv2.imshow("Masked", frame_masked)

        # ── KEYBOARD ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[EXIT] Keluar …")
            break

        elif key == ord('s'):
            if state in (STATE_IDLE, STATE_DONE):
                print("[START] Mulai scan …")
                e_current = E_MIN
                y_current = Y_HOME
                move(X=X_HOME, Y=Y_HOME, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)
                time.sleep(0.5)
                state = STATE_SCAN_E

    # ── CLEANUP ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()
