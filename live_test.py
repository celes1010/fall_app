import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import speech_recognition as sr
import requests
import csv
from datetime import datetime
import random
import os
import threading
import time

from utils.features import landmarks_to_feature_vector

# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.path.join("models", "fall_pose_model.pkl")

THRESHOLD = 0.50       # tuned threshold (you can adjust)
SMOOTH_WINDOW = 5      # moving average window
USE_VOICE = True

# Hysteresis to avoid rapid on/off toggling
RESET_FACTOR = 0.7     # we reset "fall" state when prob < THRESHOLD * RESET_FACTOR

# Cooldown between alerts (in seconds)
ALERT_COOLDOWN = 3.0   # minimum time between fall alerts

# ============ TELEGRAM CONFIG ============
TELEGRAM_BOT_TOKEN = "8430658207:AAGDkEirRcgQuenbHZ2X-1TND2d62J1UYT0"
TELEGRAM_CHAT_ID = "8291599127"
LOCATION_DESC = "Home - Living Room"
LOCATION_LAT = 8.8888
LOCATION_LON = 76.6666

LOG_FILE = "fall_events_log.csv"

# ==========================
# LOAD MODEL
# ==========================
data = joblib.load(MODEL_PATH)
if isinstance(data, dict) and "model" in data and "feature_cols" in data:
    model = data["model"]
    feature_cols = data["feature_cols"]
else:
    model = data
    feature_cols = None

n_features_model = len(feature_cols) if feature_cols is not None else None

print("[INFO] Loaded model:", MODEL_PATH)
print("[INFO] Model expects", n_features_model, "features")

# ==========================
# MEDIAPIPE SETUP
# ==========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# VOICE & SPEECH RECOGNITION SETUP (FIXED FOR THREADING)
# ==========================
recognizer = sr.Recognizer()

def speak_blocking(text):
    """Create fresh TTS engine for each call to avoid threading issues"""
    try:
        print("[VOICE OUT]:", text)
        # Create a NEW engine instance for each call
        local_engine = pyttsx3.init()
        local_engine.setProperty("rate", 150)
        local_engine.say(text)
        local_engine.runAndWait()
        # Clean up
        local_engine.stop()
        del local_engine
    except Exception as e:
        print("[WARN] TTS error:", e)

def listen_reply(timeout=10, phrase_time_limit=8) -> str:
    """Listen for user's voice response"""
    try:
        with sr.Microphone() as source:
            print("[VOICE IN]: Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[VOICE IN]: Listening (speak now)...")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
        print("[VOICE IN]: Recognizing...")
        text = recognizer.recognize_google(audio)
        print("[VOICE IN]: You said ->", text)
        return text.lower()
    except sr.WaitTimeoutError:
        print("[VOICE IN]: Timed out waiting for speech.")
        return ""
    except sr.UnknownValueError:
        print("[VOICE IN]: Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("[VOICE IN]: API error:", e)
        return ""
    except OSError as e:
        print("[VOICE IN]: Microphone error:", e)
        return ""
    except Exception as e:
        print("[VOICE IN]: Unexpected error:", e)
        return ""

# ============ DECISION LOGIC & LOGGING ============

def classify_reply(reply: str) -> str:
    """Classify user's response into categories"""
    if not reply or not reply.strip():
        return "NO_RESPONSE"

    text = reply.lower()

    help_words = [
        "help", "hurts", "pain", "emergency",
        "not okay", "cant move", "cannot move"
    ]
    ok_words = [
        "fine", "i am fine", "i'm fine",
        "okay", "ok", "all good", "no problem"
    ]

    if any(w in text for w in help_words):
        return "NEEDS_HELP"
    if any(w in text for w in ok_words):
        return "USER_OK"
    return "UNCERTAIN"

def log_event(decision: str, reply: str, prob_fall: float):
    """Log fall event to CSV file"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Simulated health data
    if decision == "NEEDS_HELP":
        heart_rate = random.randint(110, 140)
    else:
        heart_rate = random.randint(70, 100)

    if heart_rate > 120:
        health_status = "critical"
    elif heart_rate > 100:
        health_status = "elevated"
    else:
        health_status = "stable"

    row = [timestamp, decision, reply, f"{prob_fall:.2f}", heart_rate, health_status]
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "decision", "reply",
                "prob_fall", "heart_rate", "health_status"
            ])
        writer.writerow(row)

def send_telegram_alert(decision: str, reply: str, prob_fall: float):
    """Send alert via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram config missing, not sending message.")
        return

    text_lines = [
        "🚨 FALL DETECTED!",
        f"Decision: {decision}",
        f"User said: {reply or 'N/A'}",
        f"Prob. fall: {prob_fall:.2f}",
        "",
        f"Location: {LOCATION_DESC}",
        f"Coords: {LOCATION_LAT}, {LOCATION_LON}",
        "",
        "Please check on the person immediately."
    ]
    message_text = "\n".join(text_lines)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text
    }

    try:
        resp = requests.post(url, data=payload)
        if resp.status_code == 200:
            print("[INFO] Telegram alert sent.")
        else:
            print("[ERROR] Telegram API error:", resp.text)
    except Exception as e:
        print("[ERROR] Failed to send Telegram alert:", e)

def handle_alert(decision: str, reply: str, prob_fall: float):
    """Handle alert based on decision"""
    if decision in ("NEEDS_HELP", "NO_RESPONSE", "UNCERTAIN"):
        print("\n[ALERT] Fall event requires attention!")
        print("       Decision :", decision)
        print("       Reply    :", reply)
        print(f"       ProbFall : {prob_fall:.2f}\n")
        send_telegram_alert(decision, reply, prob_fall)
    else:
        print("\n[INFO] Fall detected but user said they are fine.\n")

    log_event(decision, reply, prob_fall)

# ==========================
# MAIN
# ==========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    prob_history = []
    frame_idx = 0

    # Track whether we are currently "above threshold" (fall state)
    above_threshold = False
    exit_threshold = THRESHOLD * RESET_FACTOR
    
    # Track last alert time for cooldown
    last_alert_time = 0
    
    # Track if alert was already sent for current fall episode
    alert_sent_for_current_fall = False

    # Shared state for voice interaction
    state = {
        "last_user_reply": "",
        "last_decision": "",
        "listening": False,
        "waiting_for_response": False
    }

    def ask_user_async(prob_fall):
        """Run voice interaction in background thread"""
        def worker():
            try:
                state["listening"] = True
                state["waiting_for_response"] = True
                
                speak_blocking("I detected a fall. Are you okay? Say I am fine or say help me.")
                
                reply = listen_reply()
                decision = classify_reply(reply)
                
                state["last_user_reply"] = reply
                state["last_decision"] = decision
                
                handle_alert(decision, reply, prob_fall)
            except Exception as e:
                print(f"[ERROR] Voice interaction failed: {e}")
            finally:
                state["listening"] = False
                state["waiting_for_response"] = False

        threading.Thread(target=worker, daemon=True).start()

    print("[INFO] Starting live fall detection. Press 'q' to quit.")
    print(f"[INFO] THRESHOLD={THRESHOLD}, EXIT_THRESHOLD={exit_threshold}")
    print(f"[INFO] ALERT_COOLDOWN={ALERT_COOLDOWN} seconds")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        smooth_prob = 0.0

        if results.pose_landmarks:
            lm_list = results.pose_landmarks.landmark

            # ----- feature vector -----
            feat_vec = landmarks_to_feature_vector(lm_list)

            # pad/trim safety
            if n_features_model is not None and feat_vec.shape[0] != n_features_model:
                if feat_vec.shape[0] > n_features_model:
                    feat_vec = feat_vec[:n_features_model]
                else:
                    pad = np.zeros(n_features_model - feat_vec.shape[0], dtype=np.float32)
                    feat_vec = np.concatenate([feat_vec, pad])

            X = feat_vec.reshape(1, -1)

            # ----- predict -----
            try:
                prob = float(model.predict_proba(X)[0, 1])
            except Exception:
                prob = float(model.predict(X)[0])

            prob_history.append(prob)
            if len(prob_history) > SMOOTH_WINDOW:
                prob_history.pop(0)
            smooth_prob = float(np.mean(prob_history))

            # ----- draw skeleton -----
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # ----- draw bounding box -----
            xs = [lm.x for lm in lm_list]
            ys = [lm.y for lm in lm_list]
            x_min = int(max(0, min(xs) * w))
            x_max = int(min(w - 1, max(xs) * w))
            y_min = int(max(0, min(ys) * h))
            y_max = int(min(h - 1, max(ys) * h))

            box_color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        else:
            # no pose -> push prob toward 0
            prob_history.append(0.0)
            if len(prob_history) > SMOOTH_WINDOW:
                prob_history.pop(0)
            smooth_prob = float(np.mean(prob_history))

        # ----- text overlay -----
        color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
        cv2.putText(
            frame,
            f"Fall Prob: {smooth_prob:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
        
        # Calculate time since last alert
        current_time = time.time()
        time_since_alert = current_time - last_alert_time if last_alert_time > 0 else 999
        
        cv2.putText(
            frame,
            f"Cooldown: {max(0, ALERT_COOLDOWN - time_since_alert):.1f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # User reply display
        if state["last_user_reply"]:
            cv2.putText(
                frame,
                f"User: {state['last_user_reply'][:40]}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        # Decision display
        if state["last_decision"]:
            decision_color = (0, 255, 0) if state["last_decision"] == "USER_OK" else (0, 0, 255)
            cv2.putText(
                frame,
                f"Decision: {state['last_decision']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                decision_color,
                2
            )

        # Listening indicator
        if state["listening"]:
            cv2.putText(
                frame,
                "Listening for reply...",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        # Debug info
        cv2.putText(
            frame,
            f"above_thr: {int(above_threshold)} | alert_sent: {int(alert_sent_for_current_fall)}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1
        )

        # red border when in detected fall
        if smooth_prob >= THRESHOLD:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        cv2.imshow("Fall Detection (Press Q to Quit)", frame)

        # ==========================
        # IMPROVED EDGE TRIGGER LOGIC WITH COOLDOWN
        # ==========================

        # Reset fall state when prob drops clearly below threshold
        if smooth_prob < exit_threshold:
            if above_threshold:
                print(f"[INFO] Person stood up - resetting fall detection (prob={smooth_prob:.3f})")
            above_threshold = False
            alert_sent_for_current_fall = False  # Ready for next fall

        # Trigger alert if:
        # 1. Probability is above threshold AND
        # 2. Not currently waiting for a response AND
        # 3. Haven't already sent alert for this fall episode
        if smooth_prob >= THRESHOLD:
            above_threshold = True
            
            # Only trigger if we haven't sent alert for this continuous fall episode
            if not alert_sent_for_current_fall and not state["waiting_for_response"]:
                alert_sent_for_current_fall = True
                last_alert_time = current_time
                print(f"[ALERT] NEW FALL detected at frame {frame_idx} prob={smooth_prob:.3f}")
                ask_user_async(smooth_prob)

        # quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()