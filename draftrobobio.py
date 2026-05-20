import serial
import time
import cv2
import numpy as np

# ── SERIAL ─────────────────────────────────────────────────────────
port    = '/dev/ttyUSB0'
baud    = 115200
ser     = serial.Serial(port, baud, timeout=None)
time.sleep(1)

# ── VISION CONSTANTS ───────────────────────────────────────────────
CENTER_X_MARK    = 320   # tengah frame (pixels) — sesuaikan resolusi kamera
CENTER_Y_MARK    = 240
MIN_AREA_DETECTION = 10300
MIN_AREA_PICK      = 73000
CENTER_TOLERANCE   = 25  # pixels — dianggap "sudah center" jika dalam range ini

# ── ROBOT WORKSPACE ────────────────────────────────────────────────
E_MIN, E_MAX     = 0, 200    # range rail (mm) — sesuaikan
E_STEP           = 20        # langkah scan per gerakan
X_MIN, X_MAX     = -100, 100
Z_SEARCH         = 0         # Z maksimal saat scan (posisi paling atas)
Z_PICK_HOVER     = -83       # Z hover sebelum pick
Z_PICK_DOWN      = -96       # Z saat pick
FEED_SEARCH      = 30
FEED_ALIGN       = 10

# ── STATE MACHINE ──────────────────────────────────────────────────
# SCAN_E  : gerak rail kiri-kanan, X/Y/Z tetap, cari objek
# SCAN_XY : objek tidak ketemu di semua E, geser X lalu scan E lagi
# ALIGN   : objek ketemu, koreksi X agar centroid → CENTER_X_MARK
# PICK    : objek sudah center dan area >= MIN_AREA_PICK → eksekusi pick
STATE_SCAN_E  = "SCAN_E"
STATE_SCAN_XY = "SCAN_XY"
STATE_ALIGN   = "ALIGN"
STATE_PICK    = "PICK"
STATE_DONE    = "DONE"

# ── POSISI ROBOT SAAT INI (tracking software) ──────────────────────
pos = {"X": 0.0, "Y": 216.90, "Z": Z_SEARCH, "E": E_MIN}


last_centroid = None

def send(cmd):
    ser.write((cmd + '\r').encode('utf-8'))
    while True:
        resp = ser.readline()
        if b"ok" in resp:
            break

def move(X=None, Y=None, Z=None, E=None, F=FEED_SEARCH):
    """Kirim G0 hanya axis yang berubah, update pos."""
    parts = []
    if X is not None: parts.append(f"X{X}");  pos["X"] = X
    if Y is not None: parts.append(f"Y{Y}");  pos["Y"] = Y
    if Z is not None: parts.append(f"Z{Z}");  pos["Z"] = Z
    if E is not None: parts.append(f"E{E}");  pos["E"] = E
    parts.append(f"F{F}")
    send("G0 " + " ".join(parts))

# ── FUNGSI VISION ──────────────────────────────────────────────────
def lerp(a, b, t): 
    return a + t * (b - a)

def smooth_centroid(curr):
    global last_centroid
    if last_centroid is None:
        last_centroid = curr
        return curr
    sx = int(lerp(last_centroid[0], curr[0], 0.5))
    sy = int(lerp(last_centroid[1], curr[1], 0.5))
    last_centroid = [sx, sy]
    return sx, sy

def detection(frame):
    """
    Return: (cx, cy, area, frame_debug)
    cx/cy = -1 jika tidak terdeteksi
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -- Mask warna kuning (dari kode asli kamu) --
    m1 = cv2.inRange(hsv, np.array([0,  105, 0]),   np.array([30, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([0,  105, 0]),   np.array([30, 206, 255]))
    m3 = cv2.inRange(hsv, np.array([0,   58, 117]), np.array([66, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)  # fix: chain 2-arg

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_dbg   = cv2.bitwise_and(frame, frame, mask=mask)

    best_area, cx_out, cy_out = -1, -1, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_DETECTION:
            continue
        if area > best_area:
            M  = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cx, cy   = smooth_centroid([cx, cy])
            best_area = area
            cx_out, cy_out = cx, cy

    return cx_out, cy_out, best_area, frame_dbg

# ── FUNGSI DRAW ────────────────────────────────────────────────────
def draw_plus(frame, x, y, length=20, color=(0, 255, 0)):
    cv2.line(frame, (x - length, y), (x + length, y), color, 2)
    cv2.line(frame, (x, y - length), (x, y + length), color, 2)

def draw_hud(frame, state, cx, cy, area):
    cv2.putText(frame, f"STATE: {state}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(frame, f"E:{pos['E']:.0f}  X:{pos['X']:.0f}  Z:{pos['Z']:.0f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    draw_plus(frame, CENTER_X_MARK, CENTER_Y_MARK, 30, (0, 255, 0))  # crosshair target
    if cx != -1:
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        draw_plus(frame, cx, cy, 15, (0, 0, 255))                     # centroid objek
        cv2.putText(frame, f"area:{int(area)}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inisialisasi robot
    send("G90")                              # absolute mode
    send("G28")                              # homing
    time.sleep(5)

    # Mulai dari Z maksimal, E di titik awal
    move(X=0, Y=216.90, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)

    state         = STATE_SCAN_E
    e_direction   = 1      # +1 maju, -1 mundur
    x_scan_index  = 0
    x_scan_list   = [0, 50, -50, 100, -100]  # urutan X yang dicoba saat SCAN_XY

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, area, frame_dbg = detection(frame)
        draw_hud(frame, state, cx, cy, area)

        # ── STATE: SCAN_E ─────────────────────────────────────────
        # Robot geser rail (E), X/Y/Z tetap di Z_SEARCH
        # Tujuan: temukan objek tanpa ubah posisi arm
        if state == STATE_SCAN_E:
            if cx == -1:
                # Belum ketemu — lanjut scan E
                next_e = pos["E"] + E_STEP * e_direction
                if next_e > E_MAX:
                    # Ujung kanan, balik arah
                    e_direction = -1
                    next_e = pos["E"] + E_STEP * e_direction
                elif next_e < E_MIN:
                    # Sudah scan seluruh rail, objek tidak ketemu → SCAN_XY
                    e_direction = 1
                    state = STATE_SCAN_XY
                    x_scan_index import serial
import time
import cv2
import numpy as np

# ── SERIAL ─────────────────────────────────────────────────────────
port    = '/dev/ttyUSB0'
baud    = 115200
ser     = serial.Serial(port, baud, timeout=None)
time.sleep(1)

# ── VISION CONSTANTS ───────────────────────────────────────────────
CENTER_X_MARK    = 320   # tengah frame (pixels) — sesuaikan resolusi kamera
CENTER_Y_MARK    = 240
MIN_AREA_DETECTION = 10300
MIN_AREA_PICK      = 73000
CENTER_TOLERANCE   = 25  # pixels — dianggap "sudah center" jika dalam range ini

# ── ROBOT WORKSPACE ────────────────────────────────────────────────
E_MIN, E_MAX     = 0, 200    # range rail (mm) — sesuaikan
E_STEP           = 20        # langkah scan per gerakan
X_MIN, X_MAX     = -100, 100
Z_SEARCH         = 0         # Z maksimal saat scan (posisi paling atas)
Z_PICK_HOVER     = -83       # Z hover sebelum pick
Z_PICK_DOWN      = -96       # Z saat pick
FEED_SEARCH      = 30
FEED_ALIGN       = 10

# ── STATE MACHINE ──────────────────────────────────────────────────
# SCAN_E  : gerak rail kiri-kanan, X/Y/Z tetap, cari objek
# SCAN_XY : objek tidak ketemu di semua E, geser X lalu scan E lagi
# ALIGN   : objek ketemu, koreksi X agar centroid → CENTER_X_MARK
# PICK    : objek sudah center dan area >= MIN_AREA_PICK → eksekusi pick
STATE_SCAN_E  = "SCAN_E"
STATE_SCAN_XY = "SCAN_XY"
STATE_ALIGN   = "ALIGN"
STATE_PICK    = "PICK"
STATE_DONE    = "DONE"

# ── POSISI ROBOT SAAT INI (tracking software) ──────────────────────
pos = {"X": 0.0, "Y": 216.90, "Z": Z_SEARCH, "E": E_MIN}


last_centroid = None

def send(cmd):
    ser.write((cmd + '\r').encode('utf-8'))
    while True:
        resp = ser.readline()
        if b"ok" in resp:
            break

def move(X=None, Y=None, Z=None, E=None, F=FEED_SEARCH):
    """Kirim G0 hanya axis yang berubah, update pos."""
    parts = []
    if X is not None: parts.append(f"X{X}");  pos["X"] = X
    if Y is not None: parts.append(f"Y{Y}");  pos["Y"] = Y
    if Z is not None: parts.append(f"Z{Z}");  pos["Z"] = Z
    if E is not None: parts.append(f"E{E}");  pos["E"] = E
    parts.append(f"F{F}")
    send("G0 " + " ".join(parts))

# ── FUNGSI VISION ──────────────────────────────────────────────────
def lerp(a, b, t): 
    return a + t * (b - a)

def smooth_centroid(curr):
    global last_centroid
    if last_centroid is None:
        last_centroid = curr
        return curr
    sx = int(lerp(last_centroid[0], curr[0], 0.5))
    sy = int(lerp(last_centroid[1], curr[1], 0.5))
    last_centroid = [sx, sy]
    return sx, sy

def detection(frame):
    """
    Return: (cx, cy, area, frame_debug)
    cx/cy = -1 jika tidak terdeteksi
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -- Mask warna kuning (dari kode asli kamu) --
    m1 = cv2.inRange(hsv, np.array([0,  105, 0]),   np.array([30, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([0,  105, 0]),   np.array([30, 206, 255]))
    m3 = cv2.inRange(hsv, np.array([0,   58, 117]), np.array([66, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)  # fix: chain 2-arg

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frame_dbg   = cv2.bitwise_and(frame, frame, mask=mask)

    best_area, cx_out, cy_out = -1, -1, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_DETECTION:
            continue
        if area > best_area:
            M  = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cx, cy   = smooth_centroid([cx, cy])
            best_area = area
            cx_out, cy_out = cx, cy

    return cx_out, cy_out, best_area, frame_dbg

# ── FUNGSI DRAW ────────────────────────────────────────────────────
def draw_plus(frame, x, y, length=20, color=(0, 255, 0)):
    cv2.line(frame, (x - length, y), (x + length, y), color, 2)
    cv2.line(frame, (x, y - length), (x, y + length), color, 2)

def draw_hud(frame, state, cx, cy, area):
    cv2.putText(frame, f"STATE: {state}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(frame, f"E:{pos['E']:.0f}  X:{pos['X']:.0f}  Z:{pos['Z']:.0f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    draw_plus(frame, CENTER_X_MARK, CENTER_Y_MARK, 30, (0, 255, 0))  # crosshair target
    if cx != -1:
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
        draw_plus(frame, cx, cy, 15, (0, 0, 255))                     # centroid objek
        cv2.putText(frame, f"area:{int(area)}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

def pixel_error(cx_objek, cy_objek):
    err_x = cx_objek - CENTER_X_MARK
    err_y = cy_objek - CENTER_Y_MARK

    delta_x_mm = err_x * PX_TO_MM_X
    delta_z_mm = err_y * PX_TO_MM_Z
    return delta_x_mm, delta_z_mm

# ── MAIN ───────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inisialisasi robot
    send("G90")                              # absolute mode
    send("G28")                              # homing
    time.sleep(5)

    # Mulai dari Z maksimal, E di titik awal
    move(X=0, Y=216.90, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)

    state         = STATE_SCAN_E
    e_direction   = 1      # +1 maju, -1 mundur
    x_scan_index  = 0
    x_scan_list   = [0, 50, -50, 100, -100]  # urutan X yang dicoba saat SCAN_XY

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, area, frame_dbg = detection(frame)
        draw_hud(frame, state, cx, cy, area)

        # ── STATE: SCAN_E ─────────────────────────────────────────
        # Robot geser rail (E), X/Y/Z tetap di Z_SEARCH
        # Tujuan: temukan objek tanpa ubah posisi arm
        if state == STATE_SCAN_E:
            if cx == -1:
                # Belum ketemu — lanjut scan E
                next_e = pos["E"] + E_STEP * e_direction
                if next_e > E_MAX:
                    # Ujung kanan, balik arah
                    e_direction = -1
                    next_e = pos["E"] + E_STEP * e_direction
                elif next_e < E_MIN:
                    # Sudah scan seluruh rail, objek tidak ketemu → SCAN_XY
                    e_direction = 1
                    state = STATE_SCAN_XY
                    x_scan_index = 0
                    print("[SCAN_E] Rail habis, pindah ke SCAN_XY")
                    continue
                move(E=next_e, F=FEED_SEARCH)

            else:
                # Objek ketemu di SCAN_E → masuk ALIGN
                print(f"[SCAN_E] Objek ditemukan! cx={cx} cy={cy} area={area:.0f}")
                state = STATE_ALIGN

        # ── STATE: SCAN_XY ────────────────────────────────────────
        # Geser X ke posisi berikutnya, lalu scan E lagi dari awal
        elif state == STATE_SCAN_XY:
            if x_scan_index >= len(x_scan_list):
                # Semua posisi X sudah dicoba, objek tidak ditemukan
                print("[SCAN_XY] Objek tidak ditemukan di seluruh workspace")
                state = STATE_DONE
                continue

            target_x = x_scan_list[x_scan_index]
            x_scan_index += 1
            print(f"[SCAN_XY] Coba X={target_x}, scan E dari awal")
            move(X=target_x, Z=Z_SEARCH, F=FEED_SEARCH)
            move(E=E_MIN, F=FEED_SEARCH)
            e_direction = 1
            state = STATE_SCAN_E   # kembali scan E dengan X baru

        # ── STATE: ALIGN ──────────────────────────────────────────
        # Koreksi X (dan Z jika perlu) agar centroid → CENTER_X_MARK
        # Setelah center, tunggu area >= MIN_AREA_PICK untuk lanjut pick
        elif state == STATE_ALIGN:
            if cx == -1:
                # Objek hilang saat align — kembali scan
                print("[ALIGN] Objek hilang, kembali scan")
                last_centroid = None
                state = STATE_SCAN_E
                continue

            err_x = cx - CENTER_X_MARK  # error pixel horizontal
            err_y = cy - CENTER_Y_MARK  # error pixel vertikal (Z arm)

            centered = (abs(err_x) < CENTER_TOLERANCE and
                        abs(err_y) < CENTER_TOLERANCE)

            if not centered:
                # Konversi error pixel → koreksi mm
                # Faktor 0.1 mm/pixel — kalibrasi sesuai ketinggian kamera
                px_to_mm = 0.15
                dx = -err_x * px_to_mm   # X arm berlawanan dengan X kamera
                dz =  err_y * px_to_mm   # Z turun jika objek di bawah center

                new_x = max(X_MIN, min(X_MAX, pos["X"] + dx))
                new_z = pos["Z"] + dz
                move(X=new_x, Z=new_z, F=FEED_ALIGN)

            else:
                # Sudah center, cek area
                if area >= MIN_AREA_PICK:
                    print(f"[ALIGN] Center OK, area cukup ({area:.0f}) → PICK")
                    state = STATE_PICK
                else:
                    # Turunkan Z perlahan sampai area cukup besar
                    move(Z=pos["Z"] - 5, F=FEED_ALIGN)
                    print(f"[ALIGN] Turun Z untuk perbesar area... Z={pos['Z']:.1f}")

        # ── STATE: PICK ───────────────────────────────────────────
        elif state == STATE_PICK:
            print("[PICK] Eksekusi pick & place")
            move(Z=Z_PICK_HOVER, F=FEED_SEARCH)
            move(Z=Z_PICK_DOWN,  F=6)          # F=6 = speed paling pelan
            send("M6")                          # LG3 ON (gripper/vakum)
            send("M207")
            send("G4 S1")                       # tunggu 1 detik
            move(Z=Z_SEARCH, F=FEED_SEARCH)     # angkat
            state = STATE_DONE

        # ── STATE: DONE ───────────────────────────────────────────
        elif state == STATE_DONE:
            cv2.putText(frame, "SELESAI - tekan R untuk ulang, Q keluar",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

        # ── DISPLAY ───────────────────────────────────────────────
        cv2.imshow("RNV3 Vision Control", frame)
        cv2.imshow("Mask Debug", frame_dbg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and state == STATE_DONE:
            # Reset untuk siklus baru
            last_centroid = None
            move(X=0, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)
            state       = STATE_SCAN_E
            e_direction = 1
            x_scan_index = 0
            print("[RESET] Siklus baru dimulai")

    cap.release()
    cv2.destroyAllWindows()
    send("G28")
    ser.close()

if __name__ == "__main__":
    main()= 0
                    print("[SCAN_E] Rail habis, pindah ke SCAN_XY")
                    continue
                move(E=next_e, F=FEED_SEARCH)

            else:
                # Objek ketemu di SCAN_E → masuk ALIGN
                print(f"[SCAN_E] Objek ditemukan! cx={cx} cy={cy} area={area:.0f}")
                state = STATE_ALIGN

        # ── STATE: SCAN_XY ────────────────────────────────────────
        # Geser X ke posisi berikutnya, lalu scan E lagi dari awal
        elif state == STATE_SCAN_XY:
            if x_scan_index >= len(x_scan_list):
                # Semua posisi X sudah dicoba, objek tidak ditemukan
                print("[SCAN_XY] Objek tidak ditemukan di seluruh workspace")
                state = STATE_DONE
                continue

            target_x = x_scan_list[x_scan_index]
            x_scan_index += 1
            print(f"[SCAN_XY] Coba X={target_x}, scan E dari awal")
            move(X=target_x, Z=Z_SEARCH, F=FEED_SEARCH)
            move(E=E_MIN, F=FEED_SEARCH)
            e_direction = 1
            state = STATE_SCAN_E   # kembali scan E dengan X baru

        # ── STATE: ALIGN ──────────────────────────────────────────
        # Koreksi X (dan Z jika perlu) agar centroid → CENTER_X_MARK
        # Setelah center, tunggu area >= MIN_AREA_PICK untuk lanjut pick
        elif state == STATE_ALIGN:
            if cx == -1:
                # Objek hilang saat align — kembali scan
                print("[ALIGN] Objek hilang, kembali scan")
                last_centroid = None
                state = STATE_SCAN_E
                continue

            err_x = cx - CENTER_X_MARK  # error pixel horizontal
            err_y = cy - CENTER_Y_MARK  # error pixel vertikal (Z arm)

            centered = (abs(err_x) < CENTER_TOLERANCE and
                        abs(err_y) < CENTER_TOLERANCE)

            if not centered:
                # Konversi error pixel → koreksi mm
                # Faktor 0.1 mm/pixel — kalibrasi sesuai ketinggian kamera
                px_to_mm = 0.15
                dx = -err_x * px_to_mm   # X arm berlawanan dengan X kamera
                dz =  err_y * px_to_mm   # Z turun jika objek di bawah center

                new_x = max(X_MIN, min(X_MAX, pos["X"] + dx))
                new_z = pos["Z"] + dz
                move(X=new_x, Z=new_z, F=FEED_ALIGN)

            else:
                # Sudah center, cek area
                if area >= MIN_AREA_PICK:
                    print(f"[ALIGN] Center OK, area cukup ({area:.0f}) → PICK")
                    state = STATE_PICK
                else:
                    # Turunkan Z perlahan sampai area cukup besar
                    move(Z=pos["Z"] - 5, F=FEED_ALIGN)
                    print(f"[ALIGN] Turun Z untuk perbesar area... Z={pos['Z']:.1f}")

        # ── STATE: PICK ───────────────────────────────────────────
        elif state == STATE_PICK:
            print("[PICK] Eksekusi pick & place")
            move(Z=Z_PICK_HOVER, F=FEED_SEARCH)
            move(Z=Z_PICK_DOWN,  F=6)          # F=6 = speed paling pelan
            send("M6")                          # LG3 ON (gripper/vakum)
            send("M207")
            send("G4 S1")                       # tunggu 1 detik
            move(Z=Z_SEARCH, F=FEED_SEARCH)     # angkat
            state = STATE_DONE

        # ── STATE: DONE ───────────────────────────────────────────
        elif state == STATE_DONE:
            cv2.putText(frame, "SELESAI - tekan R untuk ulang, Q keluar",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

        # ── DISPLAY ───────────────────────────────────────────────
        cv2.imshow("RNV3 Vision Control", frame)
        cv2.imshow("Mask Debug", frame_dbg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and state == STATE_DONE:
            # Reset untuk siklus baru
            last_centroid = None
            move(X=0, Z=Z_SEARCH, E=E_MIN, F=FEED_SEARCH)
            state       = STATE_SCAN_E
            e_direction = 1
            x_scan_index = 0
            print("[RESET] Siklus baru dimulai")

    cap.release()
    cv2.destroyAllWindows()
    send("G28")
    ser.close()

if __name__ == "__main__":
    main()
