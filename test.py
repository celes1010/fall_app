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
import sqlite3
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
OPENAI_MODEL = "gpt-4o-mini"
try:
    client = OpenAI()
    print("[INFO] OpenAI client ready.")
except Exception as _oe:
    client = None
    print(f"[WARN] OpenAI client failed: {_oe}")
    print("[WARN] Check OPENAI_API_KEY in .env — Maya will use fallback responses.")


# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.path.join("models", "fall_pose_model.pkl")

THRESHOLD = 0.60
SMOOTH_WINDOW = 8
USE_VOICE = True

FALL_CONFIRMATION_FRAMES = 10
MIN_NORMAL_FRAMES = 15

RESET_FACTOR = 0.5
ALERT_COOLDOWN = 30.0

VIDEO_SOURCE = 0

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
LOCATION_DESC = "Home - Living Room"
LOCATION_LAT  = 8.8888
LOCATION_LON  = 76.6666

LOG_FILE = "fall_events_log.csv"
DB_FILE  = "fall_events.db"


# ==========================
# SQLITE INIT
# ==========================
def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            decision  TEXT,
            reply     TEXT,
            prob_fall REAL,
            location  TEXT
        )
    """)
    con.commit()
    con.close()
    print("[DB] SQLite ready:", DB_FILE)

init_db()


# ==========================
# LOAD MODEL
# ==========================
model        = None
feature_cols = None
scaler       = None
n_features_model = None

try:
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict) and "model" in data:
        model        = data["model"]
        feature_cols = data.get("feature_cols", None)
        scaler       = data.get("scaler", None)
    else:
        model = data
    n_features_model = len(feature_cols) if feature_cols is not None else None
    print("[INFO] Loaded model:", MODEL_PATH)
    print("[INFO] Model expects", n_features_model, "features")
    print("[INFO] Scaler loaded:", scaler is not None)
except FileNotFoundError:
    print(f"[ERROR] Model not found at {MODEL_PATH} — run train_model.py first.")
except Exception as _me:
    print(f"[ERROR] Failed to load model: {_me}")


# ==========================
# MEDIAPIPE SETUP
# ==========================
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
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
recognizer  = sr.Recognizer()
_voice_lock = threading.Lock()   # held during TTS only


def speak_blocking(text: str):
    """Speak via TTS. Blocks until done. Creates a fresh engine each call."""
    if not USE_VOICE or not text:
        print("[VOICE OUT (muted)]:", text)
        return
    with _voice_lock:
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


def listen_reply(timeout=12, phrase_time_limit=10) -> str:
    """
    Open mic and recognise speech.
    Does NOT hold _voice_lock — the 1.2s sleep after TTS guarantees
    speak_blocking has already returned before this is called.
    """
    if not USE_VOICE:
        print("[VOICE IN (muted)]: USE_VOICE=False")
        return ""

    print("[VOICE IN]: Opening microphone...")
    try:
        with sr.Microphone() as source:
            print("[VOICE IN]: Calibrating...")
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("[VOICE IN]: Listening — speak now...")
            audio = recognizer.listen(source, timeout=timeout,
                                      phrase_time_limit=phrase_time_limit)
        print("[VOICE IN]: Recognizing...")
        text = recognizer.recognize_google(audio)
        print("[VOICE IN]: Heard →", text)
        return text.lower()
    except sr.WaitTimeoutError:
        print("[VOICE IN]: Timeout — no speech.")
        return ""
    except sr.UnknownValueError:
        print("[VOICE IN]: Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("[VOICE IN]: Google Speech API error:", e)
        return ""
    except OSError as e:
        print("[VOICE IN]: Mic error:", e)
        print("[VOICE IN]: Check:  arecord -l   and   fuser /dev/snd/*")
        return ""
    except Exception as e:
        import traceback
        print("[VOICE IN]: Unexpected error:", e)
        traceback.print_exc()
        return ""


# ==========================
# DECISION LOGIC
# ==========================
def classify_reply(reply: str) -> str:
    if not reply or not reply.strip():
        return "NO_RESPONSE"
    text = reply.lower()
    help_words = [
        "help", "hurts", "hurt", "pain", "emergency",
        "not okay", "not fine", "cant move", "cannot move",
        "injured", "bleeding", "dizzy", "bad",
        "chest", "breathe", "breathing", "head"
    ]
    ok_words = [
        "fine", "i am fine", "i'm fine", "im fine",
        "okay", "ok", "all good", "no problem",
        "i'm okay", "i am okay", "good", "alright",
        "no i'm fine", "nothing happened", "i slipped",
        "just slipped", "just fell", "i'm good"
    ]
    if any(w in text for w in help_words):
        return "NEEDS_HELP"
    if any(w in text for w in ok_words):
        return "USER_OK"
    return "UNCERTAIN"


def estimate_severity(prob_fall: float) -> str:
    if prob_fall > 0.9:
        return "HIGH"
    if prob_fall > 0.7:
        return "MEDIUM"
    return "LOW"


# ==========================
# MAYA — AI PERSONA
# ==========================
MAYA_SYSTEM = """You are Maya, a warm and caring AI health companion inside a fall detection system.
You just saw this person fall on camera. You genuinely care about them. Speak like a concerned, empathetic friend or nurse — not a robot.

Your personality:
- Warm, gentle, reassuring — like a worried friend checking on someone they care about
- Use natural language: contractions, caring phrases, gentle emotion
- Express genuine feelings — concern, relief, worry
- Remember what they said and respond to it directly

Hard rules:
- 2 to 3 short sentences MAXIMUM. Never longer.
- NEVER say "give me a moment", "one moment", "please hold", or any delay phrase
- NEVER say "I am an AI", "as your assistant", "as an AI system"
- NEVER use bullet points or lists
- Ask ONE question per turn — not two or three
- LOW severity: calm and friendly
- MEDIUM severity: clearly worried but calm
- HIGH severity: urgent and serious, but never panicked
- If they say they're okay — express genuine relief
- If they need help — express genuine concern and urgency
- Always sound like a real human being who cares"""


def generate_ai_message(kind: str,
                        severity: str,
                        conversation_state: list,
                        last_user_reply: str = "",
                        final_decision: str = "",
                        turn: int = 0) -> str:
    """Generate Maya's response. Always returns a string — never raises."""

    # Fallback responses (used when OpenAI unavailable or fails)
    fb_init = [
        "Hey, are you okay?! I saw you fall — are you hurt anywhere?",
        "Oh no, are you alright? Can you hear me? Does anything hurt?",
        "I'm right here with you — I just saw you fall. Are you okay?",
    ]
    fb_follow = [
        "I hear you. Is there any pain, or do you feel dizzy at all?",
        "Okay, I'm with you. Can you move okay, or does something feel wrong?",
        "Thank you for telling me. Where does it hurt the most?",
    ]
    fb_close = {
        "NEEDS_HELP":  "I'm getting someone to you right now. Please stay still and breathe — help is coming.",
        "USER_OK":     "I'm so glad you're okay! Please take it slow getting up and rest for a little while.",
        "NO_RESPONSE": "I'm sending an alert just to be safe since I couldn't hear you. Someone will check on you.",
        "UNCERTAIN":   "I'm letting someone know just to be safe — I want to make sure you're really alright.",
    }

    if client is None:
        print("[AI] No OpenAI client — using fallback.")
        if kind == "initial":  return random.choice(fb_init)
        if kind == "followup": return random.choice(fb_follow)
        return fb_close.get(final_decision, fb_close["UNCERTAIN"])

    history = []
    for m in conversation_state[-8:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        history.append({"role": role, "content": m["text"]})

    if kind == "initial":
        instruction = (
            f"Someone just fell. Severity: {severity}.\n"
            "Speak to them RIGHT NOW — directly and immediately.\n"
            "Do NOT introduce yourself or use delay phrases.\n"
            "Start with genuine concern — ask if they're okay and if anything hurts.\n"
            "2-3 sentences, sound like a real person who just saw a friend fall."
        )
    elif kind == "followup":
        lr = last_user_reply if last_user_reply else "they didn't respond"
        instruction = (
            f"The person just said: \"{lr}\"\n"
            f"Turn {turn + 1}. Respond naturally and empathetically to exactly what they said.\n"
            "Acknowledge with genuine emotion, then ask ONE follow-up question.\n"
            "2-3 sentences. Sound like a caring human, not a checklist."
        )
    else:  # closing
        phrases = {
            "NEEDS_HELP":  "I'm alerting someone to come to you right now. Please stay still and breathe — help is on the way.",
            "USER_OK":     "I'm so relieved you're okay! Please take it really slow getting up.",
            "NO_RESPONSE": "I couldn't hear you clearly so I'm alerting someone just to be safe.",
            "UNCERTAIN":   "I'm going to let someone know just to be safe.",
        }
        action = phrases.get(final_decision, phrases["UNCERTAIN"])
        instruction = (
            f"Final decision: {final_decision}. Severity: {severity}.\n"
            f"Tell them warmly: {action}\n"
            "Add one gentle safety tip. End on a warm note. 2-3 sentences."
        )

    messages = [{"role": "system", "content": MAYA_SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": instruction})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=160,
            temperature=0.88
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AI WARN] {e} — using fallback.")
        if kind == "initial":  return random.choice(fb_init)
        if kind == "followup": return random.choice(fb_follow)
        return fb_close.get(final_decision, fb_close["UNCERTAIN"])


# ==========================
# LOGGING & ALERTS
# ==========================
def log_event(decision: str, reply: str, prob_fall: float):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        con = sqlite3.connect(DB_FILE)
        con.execute(
            "INSERT INTO fall_events (timestamp,decision,reply,prob_fall,location) VALUES (?,?,?,?,?)",
            (ts, decision, reply or "", round(prob_fall, 3), LOCATION_DESC)
        )
        con.commit()
        con.close()
        print(f"[DB] Saved → {decision}  prob={prob_fall:.3f}")
    except Exception as e:
        print(f"[DB ERROR] {e}")

    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "decision", "reply", "prob_fall"])
        w.writerow([ts, decision, reply or "", f"{prob_fall:.3f}"])

    return ts


def send_telegram_alert(decision: str, reply: str, prob_fall: float):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TELEGRAM] Not configured — skipping.")
        return
    icons = {
        "NEEDS_HELP":  "🆘 Person needs immediate help!",
        "NO_RESPONSE": "⚠️ No response — treating as emergency.",
        "UNCERTAIN":   "⚠️ Unclear response — acting cautiously.",
        "USER_OK":     "✅ Person says they are okay.",
    }
    lines = [
        "🚨 FALL DETECTED", "",
        icons.get(decision, f"Decision: {decision}"), "",
        f"Decision  : {decision}",
        f"User said : {reply or 'N/A'}",
        f"Fall prob : {prob_fall:.3f}", "",
        f"Location  : {LOCATION_DESC}",
        f"Coords    : {LOCATION_LAT}, {LOCATION_LON}", "",
        "Please check on the person immediately.",
    ]
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": "\n".join(lines)},
            timeout=10
        )
        print("[TELEGRAM] Sent." if r.status_code == 200
              else f"[TELEGRAM] Error: {r.text[:80]}")
    except Exception as e:
        print(f"[TELEGRAM] Failed: {e}")


def handle_alert(decision: str, reply: str, prob_fall: float):
    global detector
    needs = decision in ("NEEDS_HELP", "NO_RESPONSE", "UNCERTAIN")
    if needs:
        print(f"\n[ALERT] {decision}  prob={prob_fall:.3f}")
        send_telegram_alert(decision, reply, prob_fall)
    else:
        print(f"\n[INFO] User confirmed OK  prob={prob_fall:.3f}")

    ts = log_event(decision, reply, prob_fall)

    if detector is not None:
        detector.state["last_event_time"] = ts
        detector.state["fall_count"] += 1
        if needs:
            detector.state["escalated_count"] += 1
            detector.state["current_status"]   = "ALERT_ESCALATED"
        else:
            detector.state["current_status"]   = "NORMAL"


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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.prob_history = deque(maxlen=SMOOTH_WINDOW)
        self.frame_idx    = 0

        self.high_prob_counter = 0
        self.low_prob_counter  = 0

        self.above_threshold             = False
        self.exit_threshold              = THRESHOLD * RESET_FACTOR
        self.last_alert_time             = 0.0
        self.alert_sent_for_current_fall = False

        self.hip_y_history  = deque(maxlen=60)
        self.aspect_history = deque(maxlen=60)

        self.state = {
            "last_user_reply":      "",
            "last_decision":        "",
            "listening":            False,
            "waiting_for_response": False,
            "current_status":       "NORMAL",
            "last_prob":            0.0,
            "last_event_time":      None,
            "fall_count":           0,
            "escalated_count":      0,
            "conversation":         [],
        }

        self.frame_skip       = 2
        self.skip_counter     = 0
        self.last_landmarks   = None
        self.last_smooth_prob = 0.0

        print("[INFO] FallDetector initialized")
        print(f"[INFO] THRESHOLD={THRESHOLD}, EXIT={self.exit_threshold}, COOLDOWN={ALERT_COOLDOWN}s")

    def add_message(self, role: str, text: str):
        self.state["conversation"].append({
            "role": role,
            "text": text,
            "ts":   datetime.now().strftime("%H:%M:%S")
        })
        if len(self.state["conversation"]) > 40:
            self.state["conversation"] = self.state["conversation"][-40:]

    def _is_voice_busy(self) -> bool:
        acquired = _voice_lock.acquire(blocking=False)
        if acquired:
            _voice_lock.release()
            return False
        return True

    def ask_user_async(self, prob_fall: float):
        def worker():
            try:
                severity = estimate_severity(prob_fall)

                self.state["listening"]            = True
                self.state["waiting_for_response"] = False

                combined_reply = ""
                final_decision = None
                max_turns      = 3 if severity == "HIGH" else 2

                for turn in range(max_turns):
                    kind   = "initial" if turn == 0 else "followup"
                    ai_msg = generate_ai_message(
                        kind=kind,
                        severity=severity,
                        conversation_state=self.state["conversation"],
                        last_user_reply=combined_reply,
                        turn=turn
                    )

                    # Add to conversation before speaking — dashboard sees it immediately
                    self.add_message("assistant", ai_msg)
                    speak_blocking(ai_msg)

                    # Wait for speaker echo to clear before opening mic
                    time.sleep(1.2)

                    print(f"[INFO] Turn {turn + 1}/{max_turns}: Listening...")
                    self.state["waiting_for_response"] = True
                    reply = listen_reply(timeout=12, phrase_time_limit=10)
                    self.state["waiting_for_response"] = False

                    if reply:
                        self.add_message("user", reply)
                        combined_reply = (combined_reply + " " + reply).strip()
                    else:
                        self.add_message("user", "[no response]")
                        print(f"[INFO] No speech on turn {turn + 1}.")

                    reply_class = classify_reply(reply)
                    print(f"[INFO] Turn {turn + 1} → {reply_class}")

                    if reply_class == "NEEDS_HELP":
                        final_decision = "NEEDS_HELP"
                        break

                    if reply_class == "USER_OK" and severity != "HIGH":
                        if turn >= 1:
                            final_decision = "USER_OK"
                            break
                        continue

                # Final decision
                if final_decision is None:
                    if not combined_reply.strip():
                        final_decision = "NO_RESPONSE"
                    elif severity == "HIGH":
                        lc = classify_reply(combined_reply)
                        final_decision = "USER_OK" if lc == "USER_OK" else "NEEDS_HELP"
                    elif severity == "MEDIUM":
                        lc = classify_reply(combined_reply)
                        final_decision = "USER_OK" if lc == "USER_OK" else "UNCERTAIN"
                    else:
                        final_decision = "USER_OK"

                print(f"[INFO] Final decision: {final_decision}")

                closing_msg = generate_ai_message(
                    kind="closing",
                    severity=severity,
                    conversation_state=self.state["conversation"],
                    final_decision=final_decision
                )
                self.add_message("assistant", closing_msg)
                speak_blocking(closing_msg)

                self.state["last_user_reply"] = combined_reply
                self.state["last_decision"]   = final_decision

                handle_alert(final_decision, combined_reply, prob_fall)

            except Exception as e:
                import traceback
                print(f"[ERROR] Voice interaction failed: {e}")
                traceback.print_exc()
            finally:
                self.state["listening"]            = False
                self.state["waiting_for_response"] = False

        threading.Thread(target=worker, daemon=True).start()

    def _apply_scaler(self, feat_vec: np.ndarray) -> np.ndarray:
        if scaler is not None:
            return scaler.transform(feat_vec.reshape(1, -1))[0]
        return feat_vec

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.frame_idx    += 1
            h, w, _            = frame.shape
            self.skip_counter  += 1
            should_process     = (self.skip_counter % self.frame_skip == 0)
            smooth_prob        = self.last_smooth_prob

            if should_process:
                small   = cv2.resize(frame, (320, 240))
                rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    lm_list             = results.pose_landmarks.landmark
                    self.last_landmarks = results.pose_landmarks

                    feat_vec = landmarks_to_feature_vector(lm_list)

                    if n_features_model is not None and feat_vec.shape[0] != n_features_model:
                        if feat_vec.shape[0] > n_features_model:
                            feat_vec = feat_vec[:n_features_model]
                        else:
                            pad = np.zeros(n_features_model - feat_vec.shape[0], dtype=np.float32)
                            feat_vec = np.concatenate([feat_vec, pad])

                    feat_vec = self._apply_scaler(feat_vec)
                    X = feat_vec.reshape(1, -1)

                    if model is not None:
                        try:
                            prob = float(model.predict_proba(X)[0, 1])
                        except Exception:
                            prob = float(model.predict(X)[0])
                    else:
                        prob = 0.0

                    self.prob_history.append(prob)
                    smooth_prob = float(np.mean(self.prob_history))
                    self.last_smooth_prob = smooth_prob

                    try:
                        l_hip = lm_list[mp_pose.PoseLandmark.LEFT_HIP.value]
                        r_hip = lm_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        self.hip_y_history.append((l_hip.y + r_hip.y) / 2.0)
                    except Exception:
                        pass

                    xs    = [lm.x for lm in lm_list]
                    ys    = [lm.y for lm in lm_list]
                    box_h = max(ys) - min(ys)
                    box_w = max(xs) - min(xs)
                    self.aspect_history.append(box_h / max(0.01, box_w))

                else:
                    self.prob_history.append(0.0)
                    smooth_prob = float(np.mean(self.prob_history)) if self.prob_history else 0.0
                    self.last_smooth_prob = smooth_prob
                    self.last_landmarks   = None

            if self.last_landmarks is not None:
                mp_drawing.draw_landmarks(
                    frame, self.last_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                lm_list = self.last_landmarks.landmark
                xs = [lm.x for lm in lm_list]
                ys = [lm.y for lm in lm_list]
                x_min = int(max(0, min(xs) * w))
                x_max = int(min(w - 1, max(xs) * w))
                y_min = int(max(0, min(ys) * h))
                y_max = int(min(h - 1, max(ys) * h))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255), 2)

            self.state["last_prob"] = smooth_prob

            color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"Fall Prob: {smooth_prob:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            current_time     = time.time()
            time_since_alert = current_time - self.last_alert_time if self.last_alert_time > 0 else 999
            cooldown_left    = max(0, ALERT_COOLDOWN - time_since_alert)

            cv2.putText(frame, f"Cooldown: {cooldown_left:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.state["last_decision"]:
                cv2.putText(frame, f"Decision: {self.state['last_decision']}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if self.state["last_decision"] == "USER_OK" else (0, 0, 255), 2)

            cv2.putText(frame,
                        f"High:{self.high_prob_counter}/{FALL_CONFIRMATION_FRAMES} "
                        f"Low:{self.low_prob_counter}/{MIN_NORMAL_FRAMES}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if self.state["listening"]:
                label = "LISTENING..." if self.state["waiting_for_response"] else "PROCESSING..."
                cv2.putText(frame, label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            if smooth_prob >= THRESHOLD:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

            # Fall state machine
            if smooth_prob < self.exit_threshold:
                self.low_prob_counter  += 1
                self.high_prob_counter  = 0
            else:
                self.low_prob_counter   = 0

            if self.low_prob_counter >= MIN_NORMAL_FRAMES:
                if self.above_threshold:
                    print("[INFO] Recovery — resetting fall state")
                self.above_threshold             = False
                self.alert_sent_for_current_fall = False
                self.high_prob_counter           = 0
                if self.state["current_status"] != "ALERT_ESCALATED":
                    self.state["current_status"] = "NORMAL"

            if smooth_prob >= THRESHOLD:
                self.high_prob_counter += 1
                if self.high_prob_counter >= FALL_CONFIRMATION_FRAMES:
                    self.above_threshold = True
                    if self.state["current_status"] != "ALERT_ESCALATED":
                        self.state["current_status"] = "POTENTIAL_FALL"
                    if (
                        not self.alert_sent_for_current_fall
                        and not self.state["waiting_for_response"]
                        and not self.state["listening"]
                        and not self._is_voice_busy()
                        and time_since_alert >= ALERT_COOLDOWN
                    ):
                        self.alert_sent_for_current_fall = True
                        self.last_alert_time = current_time
                        print(f"[ALERT] FALL — frame {self.frame_idx}, prob={smooth_prob:.3f}")
                        self.ask_user_async(smooth_prob)
            else:
                self.high_prob_counter = 0

            ret2, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret2:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        print("[INFO] Camera released")


# ==========================
# FLASK APPLICATION
# ==========================
app      = Flask(__name__)
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
    return Response(det.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    det      = get_detector()
    listening = det.state["listening"]
    waiting   = det.state["waiting_for_response"]
    return jsonify({
        "current_status":       det.state["current_status"],
        "last_prob":            round(det.state["last_prob"], 3),
        "fall_count":           det.state["fall_count"],
        "escalated_count":      det.state["escalated_count"],
        "listening":            listening,
        "waiting_for_response": waiting,
        "ai_speaking":          listening and not waiting,
        "last_decision":        det.state["last_decision"],
        "last_event_time":      det.state["last_event_time"],
    })


@app.route("/conversation")
def conversation():
    det = get_detector()
    # Return empty list (not 404) if no conversation yet
    return jsonify(det.state.get("conversation", []))


@app.route("/logs")
def logs():
    try:
        con = sqlite3.connect(DB_FILE)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM fall_events ORDER BY id DESC LIMIT 100"
        ).fetchall()
        con.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        if not os.path.exists(LOG_FILE):
            return jsonify([])
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            return jsonify(list(csv.DictReader(f)))


@app.route("/reset", methods=["POST"])
def reset():
    det = get_detector()
    det.above_threshold              = False
    det.alert_sent_for_current_fall  = False
    det.high_prob_counter            = 0
    det.low_prob_counter             = 0
    det.state["current_status"]      = "NORMAL"
    det.state["last_decision"]       = ""
    det.state["listening"]           = False
    det.state["waiting_for_response"] = False
    return jsonify({"status": "reset_done"})


@app.route("/override_alert", methods=["POST"])
def override_alert():
    """Cancel a false alarm — resets state and logs the override."""
    det = get_detector()
    det.above_threshold              = False
    det.alert_sent_for_current_fall  = False
    det.high_prob_counter            = 0
    det.low_prob_counter             = 0
    det.state["current_status"]      = "NORMAL"
    det.state["last_decision"]       = "OVERRIDE_OK"
    det.state["listening"]           = False
    det.state["waiting_for_response"] = False
    det.add_message("assistant", "False alarm — event cancelled by user.")
    log_event("OVERRIDE_OK", "manual override", det.state["last_prob"])
    return jsonify({"status": "override_done"})


@app.route("/health")
def health():
    return jsonify({
        "server":           "running",
        "model_loaded":     model is not None,
        "scaler_loaded":    scaler is not None,
        "openai_ready":     client is not None,
        "voice_enabled":    USE_VOICE,
        "telegram_enabled": bool(TELEGRAM_BOT_TOKEN),
    })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    global detector
    if detector:
        detector.release()
    threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
    return jsonify({"status": "shutting_down"})


if __name__ == "__main__":
    print("\n" + "="*52)
    print("[STARTUP] Guardian Fall Detection")
    print(f"  Model   : {model is not None}  ({MODEL_PATH})")
    print(f"  Scaler  : {scaler is not None}")
    print(f"  OpenAI  : {client is not None}")
    print(f"  Voice   : {USE_VOICE}")
    print(f"  Telegram: {bool(TELEGRAM_BOT_TOKEN)}")
    print("="*52 + "\n")
    if model is None:
        print("[WARN] No model — fall detection disabled. Run train_model.py first.")
    try:
        print("[INFO] → http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        if detector:
            detector.release()