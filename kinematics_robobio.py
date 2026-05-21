# kinematics_robobio.py

import math
import time
import serial

# ── SERIAL ────────────────────────────────────────────────────────
PORT    = "COM4"
BAUD    = 115200
TIMEOUT = None      # blocking – robot selalu balas "ok"

# ── WORKSPACE ─────────────────────────────────────────────────────
E_MIN  =   0
E_MAX  = 200
E_STEP =  20

Y_HOME = 216.90
Y_MIN  =  80.0
Y_MAX  = 300.0
Y_STEP =  20.0

X_HOME =   0.0
X_MIN  = -100.0
X_MAX  =  100.0

Z_SEARCH     =   0
Z_PICK_HOVER = -83
Z_PICK_DOWN  = -96

# ── KAMERA → GRIPPER OFFSET ───────────────────────────────────────
# Ukur fisik: setelah objek align ke center kamera, berapa mm capit
# masih meleset? Itu nilai offset-nya.
# Positif Y  = kamera lebih ke belakang dari ujung capit.
# Positif X  = kamera lebih ke kanan dari ujung capit.
CAMERA_OFFSET_X_MM = 0.0   # ← isi setelah kalibrasi fisik
CAMERA_OFFSET_Y_MM = 0.0   # ← isi setelah kalibrasi fisik

# ── ADAPTIVE VELOCITY – 3 ZONA ────────────────────────────────────
# Zona ditentukan dari jarak Euclidean error centroid (pixel).
#
#   dist > ZONE_FAR_PX   → FAST : objek jauh, gerak cepat
#   dist > ZONE_NEAR_PX  → MED  : mendekat, mulai presisi
#   dist ≤ ZONE_NEAR_PX  → SLOW : fine-tuning mendekati center
#
ZONE_FAR_PX  = 80    # ~12 mm dengan PX_TO_MM = 0.15
ZONE_NEAR_PX = 30    # ~4.5 mm

FEED_FAST   = 35     # mm/s
FEED_MED    = 18     # mm/s
FEED_SLOW   =  6     # mm/s
FEED_SEARCH = 30     # mm/s – sweep scan
FEED_PICK   = 15     # mm/s – gerak turun/naik pick


def adaptive_feed(err_x_px: int, err_y_px: int) -> tuple[int, str]:
    """
    Kembalikan (feed_mm_s, label_zona) berdasarkan jarak pixel dari center.
    Gunakan label untuk HUD.
    """
    dist = math.hypot(err_x_px, err_y_px)
    if dist > ZONE_FAR_PX:
        return FEED_FAST, "FAST"
    if dist > ZONE_NEAR_PX:
        return FEED_MED, "MED"
    return FEED_SLOW, "SLOW"


# ── POSISI ROBOT (tracking software) ──────────────────────────────
pos = {"X": X_HOME, "Y": Y_HOME, "Z": Z_SEARCH, "E": E_MIN}

# ── SERIAL ────────────────────────────────────────────────────────
_ser: serial.Serial | None = None


def serial_connect():
    global _ser
    _ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    time.sleep(1.5)
    print(f"[SERIAL] Terhubung ke {PORT} @ {BAUD}")


def serial_close():
    if _ser and _ser.is_open:
        _ser.close()


def send(cmd: str):
    """Kirim G/M-code dan tunggu balasan 'ok'."""
    _ser.write((cmd.strip() + "\r").encode("utf-8"))
    while True:
        if b"ok" in _ser.readline():
            break


# ── MOVE ──────────────────────────────────────────────────────────
def move(X=None, Y=None, Z=None, E=None, F=FEED_SEARCH):
    parts = []
    if X is not None: parts.append(f"X{X:.3f}"); pos["X"] = X
    if Y is not None: parts.append(f"Y{Y:.3f}"); pos["Y"] = Y
    if Z is not None: parts.append(f"Z{Z:.3f}"); pos["Z"] = Z
    if E is not None: parts.append(f"E{E:.3f}"); pos["E"] = E
    parts.append(f"F{F}")
    send("G0 " + " ".join(parts))


def dwell(seconds: float):
    send(f"G4 S{seconds}")


def home():
    send("G28")
    pos.update({"X": X_HOME, "Y": Y_HOME, "Z": Z_SEARCH, "E": E_MIN})


def gripper_open():
    send("G160")


def gripper_close():
    send("G130")