import os
from dotenv import load_dotenv

load_dotenv()  # Load .env BEFORE anything else uses env vars

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
import threading
import time
from collections import deque

from flask import Flask, Response, render_template, jsonify, request
from openai import OpenAI

from utils.features import landmarks_to_feature_vector


# ==========================
# OPENAI CLIENT
# ==========================
client = OpenAI()
OPENAI_MODEL = "gpt-4o-mini"


# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.path.join("models", "fall_pose_model.pkl")

THRESHOLD = 0.60
SMOOTH_WINDOW = 8
USE_VOICE = True

MIN_FALL_FRAMES = 12
HIP_DROP_THRESHOLD = 0.12

FALL_CONFIRMATION_FRAMES = 10
MIN_NORMAL_FRAMES = 15

RESET_FACTOR = 0.5
ALERT_COOLDOWN = 5.0

VIDEO_SOURCE = 0

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
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
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0
)


# ==========================
# VOICE & SPEECH RECOGNITION
# ==========================
recognizer = sr.Recognizer()


def speak_blocking(text: str):
    """
    Create a fresh TTS engine for each call.
    Blocks until speech is fully finished — mic only opens after this returns.
    """
    if not text:
        return
    try:
        print("[VOICE OUT]:", text)
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print("[WARN] TTS error:", e)


def listen_reply(timeout=10, phrase_time_limit=8) -> str:
    """Listen for user's voice response."""
    print("[VOICE IN]: Opening microphone...")
    try:
        with sr.Microphone() as source:
            print("[VOICE IN]: Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[VOICE IN]: Listening — speak now...")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
        print("[VOICE IN]: Audio captured, recognizing...")
        text = recognizer.recognize_google(audio)
        print("[VOICE IN]: You said ->", text)
        return text.lower()
    except sr.WaitTimeoutError:
        print("[VOICE IN]: Timed out — no speech detected.")
        return ""
    except sr.UnknownValueError:
        print("[VOICE IN]: Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("[VOICE IN]: Google Speech API error:", e)
        return ""
    except OSError as e:
        print("[VOICE IN]: Microphone not found or inaccessible:", e)
        print("[VOICE IN]: Check that a microphone is connected and not in use by another app.")
        return ""
    except Exception as e:
        import traceback
        print("[VOICE IN]: Unexpected error:", e)
        traceback.print_exc()
        return ""


# ============ DECISION LOGIC & LOGGING ============


def classify_reply(reply: str) -> str:
    """Classify user's response into categories."""
    if not reply or not reply.strip():
        return "NO_RESPONSE"

    text = reply.lower()

    help_words = [
        "help", "hurts", "hurt", "pain", "emergency",
        "not okay", "not fine", "cant move", "cannot move",
        "injured", "bleeding", "dizzy", "fell"
    ]
    ok_words = [
        "fine", "i am fine", "i'm fine", "im fine",
        "okay", "ok", "all good", "no problem",
        "i'm okay", "i am okay", "good", "alright"
    ]

    if any(w in text for w in help_words):
        return "NEEDS_HELP"
    if any(w in text for w in ok_words):
        return "USER_OK"
    return "UNCERTAIN"


def simulate_health_for_conversation(prob_fall: float):
    """Simulate heart rate & health status."""
    if prob_fall > 0.85:
        heart_rate = random.randint(118, 138)
    elif prob_fall > 0.65:
        heart_rate = random.randint(100, 120)
    else:
        heart_rate = random.randint(75, 95)

    if heart_rate > 120:
        health_status = "critical"
    elif heart_rate > 100:
        health_status = "elevated"
    else:
        health_status = "stable"

    return heart_rate, health_status


def estimate_severity(prob_fall: float, heart_rate: int) -> str:
    """Combine model probability + heart rate into LOW / MEDIUM / HIGH severity."""
    if prob_fall > 0.9 or heart_rate > 130:
        return "HIGH"
    if prob_fall > 0.7 or heart_rate > 110:
        return "MEDIUM"
    return "LOW"


def generate_ai_message(kind: str,
                        severity: str,
                        heart_rate: int,
                        health_status: str,
                        conversation_state: list,
                        last_user_reply: str = "",
                        final_decision: str = "") -> str:
    """Use OpenAI to generate an emotional, context-aware assistant message."""
    system_prompt = f"""
You are a caring, emotionally intelligent medical assistant in a fall detection system.

User health context:
- Heart rate: {heart_rate} bpm
- Health status: {health_status}
- Fall severity: {severity}

You MUST:
- Speak in SHORT, human sentences (1-3 sentences).
- Sound warm, supportive and genuinely concerned.
- For LOW severity: calm, reassuring, slightly casual.
- For MEDIUM severity: gently worried, attentive, ask more details.
- For HIGH severity: clearly serious and protective, but never panicked.
- Ask simple questions about pain, dizziness, ability to move, breathing, etc.
- Encourage the user to describe how they feel in their own words.
- Avoid robot-like phrases and repeated instructions.
"""

    history_messages = []
    for msg in conversation_state[-6:]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        history_messages.append({
            "role": role,
            "content": msg["text"]
        })

    if kind == "initial":
        user_instruction = (
            f"A fall has just been detected. Heart rate: {heart_rate} bpm, "
            f"status: {health_status}, severity: {severity}.\n"
            "Introduce yourself briefly, tell the user you noticed a possible fall, "
            "and ask them how they feel and what happened. "
            "Ask 1-2 simple questions about pain, dizziness or ability to move."
        )
    elif kind == "followup":
        if not last_user_reply:
            last_user_reply = "There was no clear response or silence."
        user_instruction = (
            f"The user's last reply was: '{last_user_reply}'.\n"
            "Respond empathetically to what they said, reflect back their concern, "
            "and then ask 1-2 follow-up questions to better understand their condition "
            "(for example: where it hurts, whether they can stand, whether they feel dizzy or short of breath). "
            "Keep it under 3 sentences."
        )
    else:  # closing
        if final_decision == "NEEDS_HELP":
            decision_phrase = "We are treating this as an emergency and alerting someone to help you."
        elif final_decision == "USER_OK":
            decision_phrase = "We are not escalating this, but you should still be careful and move slowly."
        elif final_decision == "NO_RESPONSE":
            decision_phrase = "We did not hear a clear answer, so we are acting cautiously."
        else:
            decision_phrase = "We are not completely sure, so we are being cautious."

        user_instruction = (
            f"The final decision is: {final_decision}. Heart rate: {heart_rate} bpm, "
            f"status: {health_status}, severity: {severity}.\n"
            f"Explain in friendly language what will happen next ({decision_phrase}), "
            "give one or two simple safety tips (for example, stay still, try to breathe calmly, "
            "wait for help, or call someone you trust), and reassure the user. "
            "Use at most 3 short sentences."
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_instruction})

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=220,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[WARN] OpenAI error, using fallback text:", e)
        if kind == "initial":
            return (
                "I noticed you might have fallen. I'm here with you. "
                "Please tell me how you feel right now and if anything hurts."
            )
        elif kind == "followup":
            return (
                "Thank you for telling me. Please describe where it hurts the most, "
                "or if you feel dizzy or short of breath."
            )
        else:
            if final_decision == "NEEDS_HELP":
                return (
                    "I'm going to alert someone to help you. Try to stay as comfortable as you can "
                    "and avoid sudden movements."
                )
            elif final_decision == "USER_OK":
                return (
                    "Alright, I'll mark you as safe for now. Please move slowly and rest if you feel any pain."
                )
            return "I will log this event so someone can check on you later. Please listen to your body."


def log_event(decision: str,
              reply: str,
              prob_fall: float,
              heart_rate: int = None,
              health_status: str = None):
    """Log fall event to CSV file."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if heart_rate is None:
        heart_rate = random.randint(110, 140) if decision == "NEEDS_HELP" else random.randint(70, 100)

    if health_status is None:
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

    return timestamp, heart_rate, health_status


def send_telegram_alert(decision: str, reply: str, prob_fall: float,
                        heart_rate: int = 0, health_status: str = "unknown"):
    """Send alert via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram config missing, not sending message.")
        return

    # Choose emoji based on decision
    if decision == "NEEDS_HELP":
        status_line = "🆘 Person needs immediate help!"
    elif decision == "NO_RESPONSE":
        status_line = "⚠️ No response from person — treating as emergency."
    elif decision == "UNCERTAIN":
        status_line = "⚠️ Unclear response — acting cautiously."
    else:
        status_line = "✅ Person says they are okay."

    text_lines = [
        "🚨 FALL DETECTED",
        "",
        status_line,
        "",
        f"Decision    : {decision}",
        f"User said   : {reply or 'N/A'}",
        f"Fall prob.  : {prob_fall:.2f}",
        f"Heart rate  : {heart_rate} bpm",
        f"Health      : {health_status}",
        "",
        f"Location    : {LOCATION_DESC}",
        f"Coords      : {LOCATION_LAT}, {LOCATION_LON}",
        "",
        "Please check on the person immediately."
    ]
    message_text = "\n".join(text_lines)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message_text}

    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code == 200:
            print("[INFO] Telegram alert sent.")
        else:
            print("[ERROR] Telegram API error:", resp.text)
    except Exception as e:
        print("[ERROR] Failed to send Telegram alert:", e)


def handle_alert(decision: str,
                 reply: str,
                 prob_fall: float,
                 heart_rate: int = None,
                 health_status: str = None):
    """Handle alert based on decision & update detector state."""
    global detector

    if decision in ("NEEDS_HELP", "NO_RESPONSE", "UNCERTAIN"):
        print("\n[ALERT] Fall event requires attention!")
        print("       Decision :", decision)
        print("       Reply    :", reply)
        print(f"       ProbFall : {prob_fall:.2f}\n")
        send_telegram_alert(decision, reply, prob_fall,
                            heart_rate=heart_rate or 0,
                            health_status=health_status or "unknown")
    else:
        print("\n[INFO] Fall detected but user confirmed they are fine.\n")

    ts, hr, hs = log_event(
        decision, reply, prob_fall,
        heart_rate=heart_rate,
        health_status=health_status
    )

    if detector is not None:
        detector.state["last_event_time"] = ts
        detector.state["last_health"] = {
            "heart_rate": hr,
            "health_status": hs
        }
        detector.state["fall_count"] += 1
        if decision in ("NEEDS_HELP", "NO_RESPONSE", "UNCERTAIN"):
            detector.state["escalated_count"] += 1
            detector.state["current_status"] = "ALERT_ESCALATED"
        else:
            detector.state["current_status"] = "NORMAL"


# ==========================
# FALL DETECTOR CLASS
# ==========================
class FallDetector:
    def __init__(self, source=VIDEO_SOURCE):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("[ERROR] Could not open video source.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        self.prob_history = deque(maxlen=SMOOTH_WINDOW)
        self.frame_idx = 0

        self.high_prob_counter = 0
        self.low_prob_counter = 0

        self.above_threshold = False
        self.exit_threshold = THRESHOLD * RESET_FACTOR
        self.last_alert_time = 0.0
        self.alert_sent_for_current_fall = False

        self.hip_y_history = deque(maxlen=60)
        self.aspect_history = deque(maxlen=60)

        self.state = {
            "last_user_reply": "",
            "last_decision": "",
            "listening": False,
            "waiting_for_response": False,
            "current_status": "NORMAL",
            "last_prob": 0.0,
            "last_event_time": None,
            "fall_count": 0,
            "escalated_count": 0,
            "last_health": {
                "heart_rate": 0,
                "health_status": "stable"
            },
            "conversation": []
        }

        self.frame_skip = 2
        self.skip_counter = 0

        self.last_landmarks = None
        self.last_smooth_prob = 0.0

        print("[INFO] FallDetector initialized")
        print(f"[INFO] THRESHOLD={THRESHOLD}, EXIT_THRESHOLD={self.exit_threshold}")
        print(f"[INFO] FALL_CONFIRMATION_FRAMES={FALL_CONFIRMATION_FRAMES}")
        print(f"[INFO] MIN_NORMAL_FRAMES={MIN_NORMAL_FRAMES}")

    def add_message(self, role: str, text: str):
        msg = {
            "role": role,
            "text": text,
            "ts": datetime.now().strftime("%H:%M:%S")
        }
        self.state["conversation"].append(msg)
        if len(self.state["conversation"]) > 40:
            self.state["conversation"] = self.state["conversation"][-40:]

    def ask_user_async(self, prob_fall):
        """
        Multi-turn voice conversation:
        1. AI speaks (asks if user is okay)
        2. Listens to reply
        3. Based on reply → sends Telegram alert
        """

        def worker():
            try:
                heart_rate, health_status = simulate_health_for_conversation(prob_fall)
                severity = estimate_severity(prob_fall, heart_rate)

                self.state["last_health"] = {
                    "heart_rate": heart_rate,
                    "health_status": health_status
                }
                self.state["listening"] = True
                self.state["waiting_for_response"] = False

                conv_state = self.state["conversation"]
                combined_reply = ""
                final_decision = None

                MAX_TURNS = 3

                for turn in range(MAX_TURNS):
                    if turn == 0:
                        kind = "initial"
                        last_user_reply = ""
                    else:
                        kind = "followup"
                        last_user_reply = combined_reply

                    self.state["waiting_for_response"] = False

                    # On first turn, speak immediately so user knows system detected something
                    if turn == 0:
                        speak_blocking("I noticed you might have fallen. Are you okay? Are you hurt?")

                    # Generate AI message
                    ai_msg = generate_ai_message(
                        kind=kind,
                        severity=severity,
                        heart_rate=heart_rate,
                        health_status=health_status,
                        conversation_state=conv_state,
                        last_user_reply=last_user_reply
                    )
                    self.add_message("assistant", ai_msg)

                    # Speak and BLOCK until fully done — mic opens only after this
                    speak_blocking(ai_msg)

                    # Extra pause so mic doesn't pick up TTS echo
                    time.sleep(1.0)

                    # Now listen for user reply
                    print(f"[INFO] Turn {turn + 1}: Listening for user reply...")
                    self.state["waiting_for_response"] = True
                    reply = listen_reply(timeout=12, phrase_time_limit=10)

                    self.state["waiting_for_response"] = False

                    if reply:
                        self.add_message("user", reply)
                        combined_reply = (combined_reply + " | " + reply).strip(" | ")
                    else:
                        self.add_message("user", "[no response detected]")
                        print("[INFO] No speech detected this turn.")

                    reply_class = classify_reply(reply)
                    print(f"[INFO] Reply classified as: {reply_class}")

                    # Early exit conditions
                    if reply_class == "NEEDS_HELP":
                        final_decision = "NEEDS_HELP"
                        break

                    if reply_class == "USER_OK":
                        if severity == "LOW":
                            final_decision = "USER_OK"
                            break
                        if severity == "MEDIUM" and turn >= 1:
                            final_decision = "USER_OK"
                            break
                        # HIGH severity: always ask all turns even if they say okay

                # Decide final outcome if not already decided
                if final_decision is None:
                    if not combined_reply:
                        final_decision = "NO_RESPONSE"
                    elif severity == "HIGH":
                        final_decision = "NEEDS_HELP"
                    elif severity == "MEDIUM":
                        final_decision = "UNCERTAIN"
                    else:
                        final_decision = "USER_OK"

                print(f"[INFO] Final decision: {final_decision}")

                # Speak closing message
                self.state["waiting_for_response"] = False
                closing_msg = generate_ai_message(
                    kind="closing",
                    severity=severity,
                    heart_rate=heart_rate,
                    health_status=health_status,
                    conversation_state=conv_state,
                    final_decision=final_decision
                )
                self.add_message("assistant", closing_msg)
                speak_blocking(closing_msg)

                self.state["last_user_reply"] = combined_reply
                self.state["last_decision"] = final_decision

                # === SEND TELEGRAM ALERT based on conversation outcome ===
                handle_alert(
                    final_decision,
                    combined_reply,
                    prob_fall,
                    heart_rate=heart_rate,
                    health_status=health_status
                )

            except Exception as e:
                import traceback
                print(f"[ERROR] Voice interaction failed: {e}")
                traceback.print_exc()
            finally:
                self.state["listening"] = False
                self.state["waiting_for_response"] = False

        threading.Thread(target=worker, daemon=True).start()

    def generate_frames(self):
        """Generator yielding JPEG frames."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_idx += 1
            h, w, _ = frame.shape

            self.skip_counter += 1
            should_process_pose = (self.skip_counter % self.frame_skip == 0)

            smooth_prob = self.last_smooth_prob

            if should_process_pose:
                small_frame = cv2.resize(frame, (320, 240))
                img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                if results.pose_landmarks:
                    lm_list = results.pose_landmarks.landmark
                    self.last_landmarks = results.pose_landmarks

                    feat_vec = landmarks_to_feature_vector(lm_list)

                    if n_features_model is not None and feat_vec.shape[0] != n_features_model:
                        if feat_vec.shape[0] > n_features_model:
                            feat_vec = feat_vec[:n_features_model]
                        else:
                            pad = np.zeros(n_features_model - feat_vec.shape[0], dtype=np.float32)
                            feat_vec = np.concatenate([feat_vec, pad])

                    X = feat_vec.reshape(1, -1)

                    try:
                        prob = float(model.predict_proba(X)[0, 1])
                    except Exception:
                        prob = float(model.predict(X)[0])

                    self.prob_history.append(prob)
                    smooth_prob = float(np.mean(self.prob_history))
                    self.last_smooth_prob = smooth_prob

                    # Hip tracking
                    try:
                        l_hip = lm_list[mp_pose.PoseLandmark.LEFT_HIP.value]
                        r_hip = lm_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        hip_y = (l_hip.y + r_hip.y) / 2.0
                    except Exception:
                        hip_y = 0.5
                    self.hip_y_history.append(hip_y)

                    # Aspect ratio tracking
                    xs = [lm.x for lm in lm_list]
                    ys = [lm.y for lm in lm_list]
                    x_min_f = max(0, min(xs))
                    x_max_f = min(1, max(xs))
                    y_min_f = max(0, min(ys))
                    y_max_f = min(1, max(ys))
                    box_h = y_max_f - y_min_f
                    box_w = x_max_f - x_min_f
                    aspect = box_h / max(0.01, box_w)
                    self.aspect_history.append(aspect)

                else:
                    self.prob_history.append(0.0)
                    smooth_prob = float(np.mean(self.prob_history)) if self.prob_history else 0.0
                    self.last_smooth_prob = smooth_prob
                    self.last_landmarks = None

            # Draw skeleton
            if self.last_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame,
                    self.last_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                lm_list = self.last_landmarks.landmark
                xs = [lm.x for lm in lm_list]
                ys = [lm.y for lm in lm_list]
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w - 1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h - 1, max(ys) * h))
                box_color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

            self.state["last_prob"] = smooth_prob

            # Overlays
            color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"Fall Prob: {smooth_prob:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            current_time = time.time()
            time_since_alert = current_time - self.last_alert_time if self.last_alert_time > 0 else 999

            cv2.putText(frame, f"Cooldown: {max(0, ALERT_COOLDOWN - time_since_alert):.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.state["last_decision"]:
                dec_color = (0, 255, 0) if self.state["last_decision"] == "USER_OK" else (0, 0, 255)
                cv2.putText(frame, f"Decision: {self.state['last_decision']}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dec_color, 2)

            cv2.putText(
                frame,
                f"High: {self.high_prob_counter}/{FALL_CONFIRMATION_FRAMES} | Low: {self.low_prob_counter}/{MIN_NORMAL_FRAMES}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            if self.state["listening"]:
                cv2.putText(frame, "LISTENING...", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if smooth_prob >= THRESHOLD:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

            # ===== FALL DETECTION LOGIC =====

            if smooth_prob < self.exit_threshold:
                self.low_prob_counter += 1
                self.high_prob_counter = 0
            else:
                self.low_prob_counter = 0

            if self.low_prob_counter >= MIN_NORMAL_FRAMES:
                if self.above_threshold:
                    print(f"[INFO] Sustained recovery — resetting fall state")
                self.above_threshold = False
                self.alert_sent_for_current_fall = False
                self.high_prob_counter = 0
                if self.state["current_status"] != "ALERT_ESCALATED":
                    self.state["current_status"] = "NORMAL"

            if smooth_prob >= THRESHOLD:
                self.high_prob_counter += 1

                # FIX: removed 'and is_fall_pattern' — high_prob_counter alone is sufficient
                if self.high_prob_counter >= FALL_CONFIRMATION_FRAMES:
                    self.above_threshold = True

                    if self.state["current_status"] != "ALERT_ESCALATED":
                        self.state["current_status"] = "POTENTIAL_FALL"

                    if (
                        not self.alert_sent_for_current_fall
                        and not self.state["waiting_for_response"]
                        and not self.state["listening"]
                        and (time_since_alert >= ALERT_COOLDOWN)
                    ):
                        self.alert_sent_for_current_fall = True
                        self.last_alert_time = current_time
                        print(f"[ALERT] CONFIRMED FALL at frame {self.frame_idx} "
                              f"(prob={smooth_prob:.3f}, counter={self.high_prob_counter})")
                        self.ask_user_async(smooth_prob)
            else:
                self.high_prob_counter = 0

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret2, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret2:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        print("[INFO] Camera released")


# ==========================
# FLASK APPLICATION
# ==========================
app = Flask(__name__)
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = FallDetector()
    return detector


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    det = get_detector()
    return Response(
        det.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    det = get_detector()
    return jsonify({
        "current_status": det.state["current_status"],
        "last_probability": round(det.state["last_prob"], 3),
        "fall_count": det.state["fall_count"],
        "escalated_count": det.state["escalated_count"],
        "listening": det.state["listening"],
        "waiting_for_response": det.state["waiting_for_response"],
        "last_decision": det.state["last_decision"],
        "last_health": det.state["last_health"]
    })


@app.route("/conversation")
def conversation():
    det = get_detector()
    return jsonify(det.state["conversation"])


@app.route("/logs")
def logs():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return jsonify(list(reader))


@app.route("/reset", methods=["POST"])
def reset():
    det = get_detector()
    det.above_threshold = False
    det.alert_sent_for_current_fall = False
    det.high_prob_counter = 0
    det.low_prob_counter = 0
    det.state["current_status"] = "NORMAL"
    det.state["last_decision"] = ""
    det.state["listening"] = False
    det.state["waiting_for_response"] = False
    return jsonify({"status": "reset_done"})


@app.route("/health")
def health():
    return jsonify({
        "server": "running",
        "model_loaded": model is not None,
        "voice_enabled": USE_VOICE,
        "telegram_enabled": bool(TELEGRAM_BOT_TOKEN)
    })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    det = get_detector()
    det.release()
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    return "Server shutting down..."


if __name__ == "__main__":
    try:
        print("[INFO] Fall Detection Web App Running on http://localhost:5000")
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True   # FIX: allow voice thread alongside frame thread
        )
    finally:
        if detector:
            detector.release()