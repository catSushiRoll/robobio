# vision_robobio.py

import numpy as np
import cv2

# ── DETECTION CONSTANTS ────────────────────────────────────────────
CENTER_X_MARK      = 347
CENTER_Y_MARK      = 456
MIN_AREA_DETECTION = 10300
MIN_AREA_PICK      = 73000
CENTER_TOLERANCE   = 25

PX_TO_MM_X = 0.15
PX_TO_MM_Y = 0.15

# ── COLOR PROFILES ─────────────────────────────────────────────────
# Format: { nama: [ (low_hsv, high_hsv), ... ] }
# Lebih dari satu range per warna untuk menangani variasi cahaya.
COLOR_PROFILES: dict = {
    "yellow": [
        (np.array([ 0, 100,  80]), np.array([35, 255, 255])),
        (np.array([15,  31,   0]), np.array([53, 255, 250])),
    ],
    "red": [
        (np.array([  0, 120,  70]), np.array([ 10, 255, 255])),
        (np.array([170, 120,  70]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 50]),  np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([100, 80, 50]), np.array([130, 255, 255])),
    ],
    "orange": [
        (np.array([ 5, 150, 100]), np.array([20, 255, 255])),
    ],
    "white": [
        (np.array([0,  0, 190]),  np.array([180, 40, 255])),
    ],
}

_active_profile_name: str = "yellow"
_active_profile: list     = COLOR_PROFILES["yellow"]

# ── GLOBALS ────────────────────────────────────────────────────────
last_centroid = None


# ── COLOR PROFILE API ─────────────────────────────────────────────
def set_color_profile(name: str):
    global _active_profile, _active_profile_name
    if name not in COLOR_PROFILES:
        raise ValueError(f"Profil '{name}' tidak dikenal. Pilihan: {list(COLOR_PROFILES)}")
    _active_profile_name = name
    _active_profile      = COLOR_PROFILES[name]
    print(f"[VISION] Profil warna aktif: {name.upper()}")


def get_active_profile_name() -> str:
    return _active_profile_name


def calibrate_from_roi(frame, roi_half: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """
    Sampel HSV dari kotak tengah frame. Pakai median (lebih tahan noise
    dibanding mean). Kembalikan (low_hsv, high_hsv) dengan toleransi adaptif.
    """
    fh, fw = frame.shape[:2]
    cx, cy = fw // 2, fh // 2
    roi     = frame[cy - roi_half: cy + roi_half, cx - roi_half: cx + roi_half]
    samples = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    h_med, s_med, v_med = np.median(samples, axis=0)
    low  = np.array([max(h_med - 12,   0), max(s_med - 45,  20), max(v_med - 55,  20)], np.uint8)
    high = np.array([min(h_med + 12, 180), min(s_med + 45, 255), min(v_med + 55, 255)], np.uint8)
    return low, high


def save_calibrated_profile(low: np.ndarray, high: np.ndarray, name: str = "custom"):
    COLOR_PROFILES[name] = [(low, high)]
    set_color_profile(name)


def draw_calib_roi(frame, roi_half: int = 28):
    fh, fw = frame.shape[:2]
    cx, cy = fw // 2, fh // 2
    cv2.rectangle(frame,
                  (cx - roi_half, cy - roi_half),
                  (cx + roi_half, cy + roi_half),
                  (0, 200, 255), 2)
    cv2.putText(frame, "ROI CALIB", (cx - roi_half, cy - roi_half - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return frame


# ── INTERNAL ──────────────────────────────────────────────────────
def _build_mask(hsv_frame: np.ndarray) -> np.ndarray:
    combined = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for low, high in _active_profile:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv_frame, low, high))
    return combined


# ── DRAW HELPERS ──────────────────────────────────────────────────
def draw_plus(frame, x, y, length, color):
    cv2.line(frame, (x - length, y), (x + length, y), color, 2)
    cv2.line(frame, (x, y - length), (x, y + length), color, 2)


def lerp(a, b, t):
    return a + t * (b - a)


# ── MATH ──────────────────────────────────────────────────────────
def pixel_to_mm(cx_obj: int, cy_obj: int) -> tuple[float, float]:
    return (cx_obj - CENTER_X_MARK) * PX_TO_MM_X, \
           (cy_obj - CENTER_Y_MARK) * PX_TO_MM_Y


def pixel_error(cx_obj: int, cy_obj: int) -> tuple[int, int]:
    """Error mentah dalam pixel, untuk adaptive_feed."""
    return cx_obj - CENTER_X_MARK, cy_obj - CENTER_Y_MARK


# ── DETECTION ─────────────────────────────────────────────────────
def detection(frame: np.ndarray):
    """
    Deteksi objek dengan warna profil aktif.

    Return:
        (cx, cy, contour_width, frame_color, x1, y1, x2, y2)
        Semua -1 jika tidak ada objek.
    """
    global last_centroid

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _build_mask(hsv)

    kernel   = np.ones((3, 3), np.uint8)
    opening  = cv2.morphologyEx(mask,    cv2.MORPH_OPEN,  kernel)
    closing  = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing, kernel, iterations=1)

    frame_color = cv2.bitwise_and(frame, frame, mask=dilation)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA_DETECTION:
            break

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if last_centroid is None:
            last_centroid = [cx, cy]
        else:
            last_centroid[0] = int(lerp(last_centroid[0], cx, 0.5))
            last_centroid[1] = int(lerp(last_centroid[1], cy, 0.5))

        return (last_centroid[0], last_centroid[1],
                max(x + w, y + h), frame_color,
                x, y, x + w, y + h)

    last_centroid = None
    return -1, -1, -1, frame_color, -1, -1, -1, -1


def draw_overlay(frame, cx, cy):
    draw_plus(frame, CENTER_X_MARK, CENTER_Y_MARK, 15, (255, 0, 0))
    if cx != -1 and cy != -1:
        draw_plus(frame, cx, cy, 10, (0, 255, 0))
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame


# ── STANDALONE TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    set_color_profile("yellow")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, _, _, x1, y1, x2, y2 = detection(frame)
        frame = draw_overlay(frame, cx, cy)

        if cx != -1:
            dx, dy = pixel_to_mm(cx, cy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"dx={dx:.2f}mm dy={dy:.2f}mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "TIDAK ADA OBJEK",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Vision Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()