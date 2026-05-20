import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Color Parameters')
cv2.createTrackbar('L-H', 'Color Parameters', 0, 179, nothing)
cv2.createTrackbar('L-S', 'Color Parameters', 0, 255, nothing)
cv2.createTrackbar('L-V', 'Color Parameters', 0, 255, nothing)
cv2.createTrackbar('U-H', 'Color Parameters', 179, 179, nothing)
cv2.createTrackbar('U-S', 'Color Parameters', 255, 255, nothing)
cv2.createTrackbar('U-V', 'Color Parameters', 255, 255, nothing)

def get_color_mask(frame):
    l_h = cv2.getTrackbarPos('L-H', 'Color Parameters')
    l_s = cv2.getTrackbarPos('L-S', 'Color Parameters')
    l_v = cv2.getTrackbarPos('L-V', 'Color Parameters')
    u_h = cv2.getTrackbarPos('U-H', 'Color Parameters')
    u_s = cv2.getTrackbarPos('U-S', 'Color Parameters')
    u_v = cv2.getTrackbarPos('U-V', 'Color Parameters')

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)

    # blur = cv2.GaussianBlur(mask, (5, 5), 0)
    blur_kernel = 3
    blur = cv2.medianBlur(mask, blur_kernel)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dilation = cv2.dilate(closing, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frame_color = cv2.bitwise_and(frame, frame, mask=dilation)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Area of object detected: {cv2.contourArea(largest_contour)}")

    return frame_color, mask

if __name__ == '__main__':
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color_masked_frame, mask = get_color_mask(frame)
        cv2.imshow('Detected Frame', color_masked_frame)
        cv2.imshow('Masked Frame', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
