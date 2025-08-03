import cv2;
import numpy as np;
import time;
import threading;
from pygame import mixer;
from collections import deque;

# ======================
# CONFIGURATION
# ======================
STRESS_THRESHOLD = 0.65;        # Base threshold (0-1)
CARDIAC_THRESHOLD = 0.55;       # For chest-focused cases
ALERT_COOLDOWN = 30;            # Seconds between alerts
MIN_CONSECUTIVE_FRAMES = 8;     # Frames needed for detection
MIN_FACE_SIZE = 150;            # Minimum face size (pixels)

# Condition weights (sum to 1.0)
WEIGHTS = {
    'facial': 0.5,
    'shivering': 0.3, 
    'chest': 0.2
};

# ======================
# INITIAL SETUP 
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
);

alarm_active = False;
alarm_color = (0, 0, 0);
last_alert_time = 0;
prev_face_roi = None;
stress_frames = deque(maxlen=MIN_CONSECUTIVE_FRAMES);

# ======================
# IMPROVED STRESS DETECTION
# ======================
def calculate_stress_level(facial, shivering, chest):
    # Weighted combined score
    score = (facial * WEIGHTS['facial'] + 
             shivering * WEIGHTS['shivering'] + 
             chest * WEIGHTS['chest']);
    
    # Special case for cardiac patterns
    if chest > 0.7 and facial > 0.6:
        return score >= CARDIAC_THRESHOLD;
    return score >= STRESS_THRESHOLD;

def detect_stress(face_roi):
    global prev_face_roi;

    # Convert to grayscale and resize
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY);
    gray = cv2.resize(gray, (64, 64));

    # 1. Facial stress (edges + wrinkles)
    edges = cv2.Canny(gray, 30, 100);
    edge_density = np.sum(edges) / (edges.size * 255);
    
    # 2. Mouth tension
    mouth_roi = gray[40:60, 15:50];
    _, mouth_thresh = cv2.threshold(mouth_roi, 100, 255, cv2.THRESH_BINARY_INV);
    mouth_tension = np.sum(mouth_thresh) / (mouth_thresh.size * 255);
    
    # 3. Shivering detection
    shivering = False;
    if prev_face_roi is not None:
        prev_resized = cv2.resize(prev_face_roi, (64, 64));
        diff = cv2.absdiff(gray, prev_resized);
        shivering = np.mean(diff) > 8;
    
    prev_face_roi = gray.copy();
    
    # Return individual metrics
    facial_score = min(edge_density * 1.5 + (1 - mouth_tension), 1.0);
    return facial_score, shivering;

# ======================
# CHEST MOTION DETECTION
# ====================== 
def detect_chest_motion(chest_region):
    if chest_region.size == 0:
        return 0.0;
    
    # Optical flow based motion
    prev = cv2.GaussianBlur(chest_region, (5,5), 0);
    flow = cv2.calcOpticalFlowFarneback(
        prev, cv2.GaussianBlur(chest_region, (5,5), 0), 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    );
    motion = np.sqrt(flow[...,0]**2 + flow[...,1]**2);
    return min(np.mean(motion) / 5.0, 1.0);  # Normalized to 0-1

# ======================
# ALERT SYSTEM
# ======================
def trigger_alert():
    global alarm_active, last_alert_time;
    if time.time() - last_alert_time < ALERT_COOLDOWN:
        return;
    alarm_active = True;
    last_alert_time = time.time();
    threading.Thread(target=glow_bulb, daemon=True).start();
    threading.Thread(target=play_siren, args=("siren.mp3",), daemon=True).start();

def glow_bulb():
    global alarm_color;
    while alarm_active:
        for intensity in range(0, 255, 5):
            alarm_color = (intensity, intensity // 2, 0);
            time.sleep(0.02);
        for intensity in range(255, 0, -5):
            alarm_color = (intensity, intensity // 2, 0);
            time.sleep(0.02);

def play_siren(file_path):
    mixer.init();
    mixer.music.load(file_path);
    mixer.music.play();
    while mixer.music.get_busy():
        time.sleep(0.1);

# ====================== 
# MAIN LOOP
# ======================
def main():
    global alarm_active, stress_frames;

    cap = cv2.VideoCapture(0);
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);

    while True:
        ret, frame = cap.read();
        if not ret: 
            break;
            
        frame = cv2.flip(frame, 1);
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, 
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        );
        
        current_stress = False;
        condition_text = "Normal";
        
        for (x, y, w, h) in faces:
            # Get regions of interest
            face_roi = frame[y:y+h, x:x+w];
            chest_region = gray[y+h:y+int(1.3*h), x:x+w];
            
            # Detect signals
            facial, shivering = detect_stress(face_roi);
            chest = detect_chest_motion(chest_region);
            
            # Determine stress
            stress_detected = calculate_stress_level(facial, shivering, chest);
            
            # Classify condition
            if stress_detected:
                if chest > 0.7 and facial > 0.6:
                    condition_text = "Cardiac Pattern";
                elif shivering and facial > 0.7:
                    condition_text = "Panic/Shivering"; 
                else:
                    condition_text = "General Stress";
                current_stress = True;
            
            # Display metrics
            cv2.rectangle(frame, (x,y), (x+w,y+h), 
                         (0,0,255) if stress_detected else (0,255,0), 2);
            
            cv2.putText(frame, f"Facial: {facial:.0%}", (x, y-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
            cv2.putText(frame, f"Shiver: {'Yes' if shivering else 'No'}", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
            cv2.putText(frame, f"Chest: {chest:.0%}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1);
            cv2.putText(frame, condition_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if stress_detected else (0,255,0), 1);

        # Alert logic
        stress_frames.append(1 if current_stress else 0);
        if sum(stress_frames) >= MIN_CONSECUTIVE_FRAMES and not alarm_active:
            trigger_alert();
        elif not current_stress and alarm_active:
            alarm_active = False;
            
        # Display
        if alarm_active:
            cv2.circle(frame, (50,50), 30, alarm_color, -1);
            cv2.putText(frame, "ALERT!", (90,55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, alarm_color, 2);
        
        cv2.imshow("Patient Monitoring", frame);
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    cap.release();
    cv2.destroyAllWindows();

# Run the system
if __name__ == "__main__":
    main();
