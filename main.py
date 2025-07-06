import cv2
import numpy as np
import time
import threading
from pygame import mixer

# ======================
# CONFIGURATION
# ======================
STRESS_THRESHOLD = 0.7       # 0-1 sensitivity
ALERT_COOLDOWN = 30          # Seconds between alerts

# ======================
# INITIAL SETUP
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

alarm_active = False
alarm_color = (0, 0, 0)
last_alert_time = 0
prev_face_roi = None

# ======================
# STRESS DETECTION FUNCTION
# ======================
def detect_stress(face_roi):
    global prev_face_roi

    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Resize to consistent size
    gray = cv2.resize(gray, (48, 48))

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (48 * 48 * 255)

    # Mouth openness
    mouth_roi = gray[30:48, 10:38]
    mouth_openness = np.mean(mouth_roi)

    # Shivering detection
    shivering = False
    if prev_face_roi is not None:
        diff = cv2.absdiff(gray, prev_face_roi)
        diff_mean = np.mean(diff)
        if diff_mean > 10:
            shivering = True

    # Store current for next frame
    prev_face_roi = gray.copy()

    # Stress score
    stress_score = min(edge_density * 2 + (1 - mouth_openness / 255), 1)

    return shivering, stress_score > STRESS_THRESHOLD

# ======================
# VISUAL ALARM FUNCTION
# ======================
def glow_bulb():
    global alarm_color
    while alarm_active:
        for intensity in range(0, 255, 5):
            alarm_color = (intensity, intensity // 2, 0)
            time.sleep(0.02)
        for intensity in range(255, 0, -5):
            alarm_color = (intensity, intensity // 2, 0)
            time.sleep(0.02)

# ======================
# AUDIO ALERT FUNCTION (pygame)
# ======================
def play_siren(file_path):
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)

# ======================
# MAIN FUNCTION
# ======================
def main():
    global alarm_active, last_alert_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Silent Aid - Patient Monitoring", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        stress_detected = False

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            shivering, stress_metric = detect_stress(face_roi)

            # Heart touch detection
            chest_region_y1 = y + h
            chest_region_y2 = y + int(1.5 * h)
            chest_region = gray[chest_region_y1:chest_region_y2, x:x + w]
            hand_near_heart = False

            if chest_region.size > 0:
                motion = cv2.absdiff(chest_region, cv2.GaussianBlur(chest_region, (9,9),0))
                motion_level = np.mean(motion)
                if motion_level > 2.5:
                    hand_near_heart = True

            # Combine BOTH conditions
            if shivering and hand_near_heart:
                stress_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "STRESS DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = time.time()
        if stress_detected and not alarm_active and (current_time - last_alert_time > ALERT_COOLDOWN):
            alarm_active = True
            last_alert_time = current_time

            threading.Thread(target=glow_bulb, daemon=True).start()
            threading.Thread(target=play_siren, args=("siren.mp3",), daemon=True).start()

        if not stress_detected and alarm_active:
            alarm_active = False

        if alarm_active:
            cv2.circle(frame, (50, 50), 30, alarm_color, -1)
            cv2.putText(frame, "ALERT!", (90, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, alarm_color, 2)

        cv2.putText(frame, "Room: 404", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Silent Aid - Patient Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
