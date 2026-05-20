import serial
import time
import numpy as np
import cv2

# Constant for picking
CENTER_X_MARK      = 320    # di-set ulang via klik kiri mouse
CENTER_Y_MARK      = 240
MIN_AREA_DETECTION = 10300
MIN_AREA_PICK      = 73000
CENTER_TOLERANCE   = 25  

PX_TO_MM_X = 0.15           # mm per pixel, sumbu X robot
PX_TO_MM_Y = 0.15           # mm per pixel, sumbu Y robot

# ── ROBOT WORKSPACE ────────────────────────────────────────────────
E_MIN         = 0
E_MAX         = 200
E_STEP        = 20
X_MIN         = -100
X_MAX         = 100
Z_SEARCH      = 0           # Z maksimal saat scan
Z_PICK_HOVER  = -83
Z_PICK_DOWN   = -96
FEED_SEARCH   = 30
FEED_ALIGN    = 10

# ── STATES ─────────────────────────────────────────────────────────
STATE_IDLE    = "IDLE"      # tunggu user tekan S untuk mulai
STATE_SCAN_E  = "SCAN_E"
STATE_SCAN_XY = "SCAN_XY"
STATE_ALIGN   = "ALIGN"
STATE_PICK    = "PICK"
STATE_CALIB   = "CALIB"
STATE_DONE    = "DONE"

# ── POSISI ROBOT (tracking software) ───────────────────────────────
pos = {"X": 0.0, "Y": 216.90, "Z": Z_SEARCH, "E": E_MIN}

# ── GLOBALS ────────────────────────────────────────────────────────
last_centroid  = None
calib_step     = 0          # langkah prosedur kalibrasi (0/1/2)
calib_cx1      = None   
last_centroid_pd = None 

LOW_HSV_YELLOW1 = np.array([0, 105, 0])
HIGH_HSV_YELLOW1 = np.array([30, 255, 255])

LOW_HSV_YELLOW2= np.array([0, 105, 0])
HIGH_HSV_YELLOW2 = np.array([30, 206, 255])

LOW_HSV_YELLOW3 = np.array([0, 58, 117])
HIGH_HSV_YELLOW3 = np.array([66, 255, 255])


port = 'COM4' # Lihat nama port dalam Arduino IDE: Tools > Port:
baud = 115200 # Default 115200
timeout = None # Biarkan seperti ini
ser = serial.Serial(port,baud,timeout=timeout)
time.sleep(1)

def send (cmd):
    ser.write((cmd+'\r').encode('utf-8'))
    while True:
        respond = ser.readline()
        if b"ok" in respond:
            break

def draw_plus(frame, x, y, len, color):
    cv2.line(frame, (x-len, y), (x+len, y), color, 2)
    cv2.line(frame, (x, y-len), (x, y+len), color, 2)

def lerp(start_p, end_p, alpha):
    return start_p + (alpha*(end_p - start_p))

def lerped_cent(last_p, curr_p): # last_p = startnya, curr_p = targetnya
    # alpha = 0.07 # faktor pengali - determines the speed
    alpha = 0.5
    new_x = (lerp(last_p[0], curr_p[0], alpha))
    new_y = (lerp(last_p[1], curr_p[1], alpha))
    # print((curr_p[0]-last_p[0]) + (curr_p[1]-last_p[1]))
    return int(new_x), int(new_y)

def calculate_crosshair(camera_position, altitude):
    # Define the tic-tac-toe grid positions for the camera
    grid_positions = {
        'left-up': (160, 120),
        'up': (252, 236),
        'right-up': (480, 120),
        'left': (160, 240),
        'center': (210, 185),
        'right': (480, 240),
        'left-down': (160, 360),
        'down': (242, 35),
        'right-down': (480, 360)
    }

    # Opposite corners for each camera position
    opposite_positions = {
        'left-up': (480, 360),
        'up': (252, 236),
        'right-up': (160, 360),
        'left': (480, 240),
        'center': (210, 185),
        'right': (160, 240),
        'left-down': (480, 120),
        'down': (242, 35),
        'right-down': (160, 120)
    }

def pixel_to__mm(cx_obj, cy_obj):
    err_x_px= cx_obj - CENTER_X_MARK
    err_y_px= cy_obj - CENTER_Y_MARK
    delta_x=err_x_px * PX_TO_MM_X
    delta_y=err_y_px * PX_TO_MM_Y
    return delta_x, delta_y

def detection (frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOW_HSV_YELLOW1, HIGH_HSV_YELLOW1)
    mask2 = cv2.inRange(hsv, LOW_HSV_YELLOW2, HIGH_HSV_YELLOW2)
    mask3 = cv2.inRange(hsv, LOW_HSV_YELLOW3, HIGH_HSV_YELLOW3)
    mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)

    kernel = np.ones((3,3), np.uint8)
    # blur = cv2.medianBlur(frame, blur_kernel)
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dilation = cv2.dilate(closing, kernel, iterations=1)

    frame_color = cv2.bitwise_and(frame, frame, mask=dilation)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_area = -1
    cx_target = -1
    cy_target = -1
    approx_arr_target = []
    contour_width_target = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA_DETECTION < area:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # x,y,w,h = cv2.boundingRect(cnt)
            # contour_width = max(x+w, y+h)
            # if area > biggest_area:
            #     current_centroid = [cx, cy]
            #     if current_centroid and (last_centroid_pd is None):
            #         last_centroid_pd = current_centroid
            #     else:
            #         current_centroid[0], current_centroid[1] = lerped_cent(last_centroid_pd,current_centroid)
            #         last_centroid_pd = current_centroid
            #         cx, cy = current_centroid[0], current_centroid[1]
                    
            #     biggest_area = area
            #     cx_target = cx
            #     cy_target = cy
            #     # detected_contour = True
            #     contour_width_target = contour_width

            if last_centroid is None:
                last_centroid = [cx, cy]
                return cx, cy
            sx = int (lerp(last_centroid[0], cx, 0.5))
            sy = int (lerp(last_centroid[1], cy, 0.5))
            last_centroid = [sx,sy]
            cx_target = last_centroid[0]
            cy_target = last_centroid[1]
    return cx_target, cy_target, contour_width_target, frame_color

def move(X=None, Y=None, Z=None, E=None, F=FEED_SEARCH):
    parts = []
    if X is not None: parts.append(f"X{X}");  pos["X"] = X
    if Y is not None: parts.append(f"Y{Y}");  pos["Y"] = Y
    if Z is not None: parts.append(f"Z{Z}");  pos["Z"] = Z
    if E is not None: parts.append(f"E{E}");  pos["E"] = E
    parts.append(f"F{F}")
    send("G0 " + " ".join(parts))



if __name__ == "__main__":
    