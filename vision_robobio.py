import numpy as np
import cv2

# ── DETECTION CONSTANTS ────────────────────────────────────────────
CENTER_X_MARK      = 320
CENTER_Y_MARK      = 240
MIN_AREA_DETECTION = 10300
MIN_AREA_PICK      = 73000
CENTER_TOLERANCE   = 25

PX_TO_MM_X = 0.15           # mm per pixel, sumbu X robot
PX_TO_MM_Y = 0.15           # mm per pixel, sumbu Y robot

# ── HSV COLOR RANGES ───────────────────────────────────────────────
LOW_HSV_YELLOW1  = np.array([0, 105, 0])
HIGH_HSV_YELLOW1 = np.array([30, 255, 255])

LOW_HSV_YELLOW2  = np.array([0, 105, 0])
HIGH_HSV_YELLOW2 = np.array([30, 206, 255])

LOW_HSV_YELLOW3  = np.array([0, 58, 117])
HIGH_HSV_YELLOW3 = np.array([66, 255, 255])

# ── GLOBALS ────────────────────────────────────────────────────────
last_centroid    = None
last_centroid_pd = None


def draw_plus(frame, x, y, length, color):
    cv2.line(frame, (x - length, y), (x + length, y), color, 2)
    cv2.line(frame, (x, y - length), (x, y + length), color, 2)


def lerp(start_p, end_p, alpha):
    return start_p + (alpha * (end_p - start_p))


def lerped_cent(last_p, curr_p):
    alpha = 0.5
    new_x = lerp(last_p[0], curr_p[0], alpha)
    new_y = lerp(last_p[1], curr_p[1], alpha)
    return int(new_x), int(new_y)


def pixel_to_mm(cx_obj, cy_obj):
    err_x_px = cx_obj - CENTER_X_MARK
    err_y_px = cy_obj - CENTER_Y_MARK
    delta_x  = err_x_px * PX_TO_MM_X
    delta_y  = err_y_px * PX_TO_MM_Y
    return delta_x, delta_y


def detection(frame):
    global last_centroid

    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOW_HSV_YELLOW1, HIGH_HSV_YELLOW1)
    mask2 = cv2.inRange(hsv, LOW_HSV_YELLOW2, HIGH_HSV_YELLOW2)
    mask3 = cv2.inRange(hsv, LOW_HSV_YELLOW3, HIGH_HSV_YELLOW3)
    mask  = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask,    cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dilation = cv2.dilate(closing, kernel, iterations=1)

    frame_color = cv2.bitwise_and(frame, frame, mask=dilation)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx_target           = -1
    cy_target           = -1
    contour_width_target = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA_DETECTION:
            M  = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if last_centroid is None:
                last_centroid = [cx, cy]
                return cx, cy, contour_width_target, frame_color

            sx = int(lerp(last_centroid[0], cx, 0.5))
            sy = int(lerp(last_centroid[1], cy, 0.5))
            last_centroid = [sx, sy]
            cx_target     = last_centroid[0]
            cy_target     = last_centroid[1]

    return cx_target, cy_target, contour_width_target, frame_color


def draw_overlay(frame, cx, cy):
    """Gambar crosshair target dan marker tengah di frame."""
    draw_plus(frame, CENTER_X_MARK, CENTER_Y_MARK, 15, (255, 0, 0))   # biru = center kamera
    if cx != -1 and cy != -1:
        draw_plus(frame, cx, cy, 10, (0, 255, 0))                      # hijau = objek terdeteksi
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, width, frame_color = detection(frame)
        frame = draw_overlay(frame, cx, cy)

        if cx != -1:
            dx, dy = pixel_to_mm(cx, cy)
            cv2.putText(frame, f"cx={cx} cy={cy}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"dx={dx:.2f}mm dy={dy:.2f}mm", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Vision", frame)
        cv2.imshow("Masked", frame_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()