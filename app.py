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
# CONFIG — TUNED TO REDUCE FALSE POSITIVES
# ==========================
MODEL_PATH = os.path.join("models", "fall_pose_model.pkl")

# ── Detection thresholds ──
THRESHOLD = 0.70                # FIX: raised from 0.60 → 0.70 (reduces false triggers)
SMOOTH_WINDOW = 10              # FIX: increased from 8 → 10 (smoother probability curve)

# ── Fall confirmation ──
FALL_CONFIRMATION_FRAMES = 15   # FIX: raised from 10 → 15 (needs more sustained high-prob)
MIN_NORMAL_FRAMES = 20          # FIX: raised from 15 → 20 (needs longer recovery to reset)

# ── Geometric gating (RE-ENABLED — this was the main fix) ──
HIP_DROP_THRESHOLD = 0.10       # hip must drop by at least 10% of frame height
HIP_DROP_WINDOW = 20            # look back this many frames for hip drop
ASPECT_RATIO_THRESHOLD = 1.2    # NEW: bounding box must be wider than tall (fallen = < 1.2)

# ── Hysteresis & cooldown ──
RESET_FACTOR = 0.5
ALERT_COOLDOWN = 45.0           # FIX: raised from 30 → 45s (voice interaction takes time)

# ── Voice ──
USE_VOICE = True

# ── Video ──
VIDEO_SOURCE = 0

# ── Telegram ──
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
_voice_lock = threading.Lock()

# ── Detect female voice ID once at startup, reuse it for every engine ──
_female_voice_id = None

def _detect_female_voice():
    """Probe available voices ONCE and store the female voice ID."""
    global _female_voice_id
    try:
        tmp = pyttsx3.init()
        voices = tmp.getProperty("voices")

        # Try known female voice names across platforms
        for v in voices:
            name = (v.name or "").lower()
            vid  = (v.id or "").lower()
            if any(kw in name for kw in ["female", "zira", "samantha",
                                          "victoria", "hazel", "susan"]):
                _female_voice_id = v.id
                print(f"[TTS] Female voice found: {v.name} → {v.id}")
                break
            if "+f" in vid or "female" in vid:
                _female_voice_id = v.id
                print(f"[TTS] Female voice found (by ID): {v.id}")
                break

        # espeak fallback
        if _female_voice_id is None:
            _female_voice_id = "english+f3"
            print(f"[TTS] No named female voice — using espeak fallback: english+f3")

        print(f"[TTS] Available voices: {[(v.name, v.id) for v in voices]}")
        tmp.stop()
        del tmp
    except Exception as e:
        print(f"[TTS] Voice detection failed: {e}")
        _female_voice_id = None

if USE_VOICE:
    _detect_female_voice()


def speak_blocking(text: str):
    """
    Speak text via pyttsx3. Creates a FRESH engine each call.
    
    Why fresh engine? pyttsx3.runAndWait() uses an internal event loop that
    deadlocks when reused across threads. Fresh engine = reliable every time.
    The female voice ID is cached so there's no extra overhead.
    
    Includes a watchdog: if TTS hangs for >15s, the call is abandoned.
    """
    if not USE_VOICE or not text:
        print("[VOICE OUT (muted)]:", text)
        return

    _done = threading.Event()

    def _tts_inner():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.setProperty("volume", 1.0)
            if _female_voice_id:
                try:
                    engine.setProperty("voice", _female_voice_id)
                except Exception:
                    pass  # voice ID might not exist on this system
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"[TTS WARN] {e}")
        finally:
            _done.set()

    with _voice_lock:
        print("[VOICE OUT]:", text)
        t = threading.Thread(target=_tts_inner, daemon=True)
        t.start()
        # Watchdog — if TTS hangs, give up after 15 seconds
        if not _done.wait(timeout=15):
            print("[TTS ERROR] Speech timed out after 15s — skipping")
        # Small extra wait for audio device to fully release
        time.sleep(0.1)


def listen_reply(timeout=12, phrase_time_limit=10) -> str:
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
# MAYA — AI PERSONA (rewritten for natural speech)
# ==========================
MAYA_SYSTEM = """You are Maya, a young woman who works as a home care aide. You're monitoring someone through a fall detection camera. You just saw them fall.

You talk like a REAL person — think how a 28-year-old nurse would actually talk if she saw her patient fall. Use contractions. Use filler words occasionally ("oh gosh", "okay okay"). React emotionally.

CRITICAL SPEECH RULES (this will be spoken aloud via text-to-speech, NOT read):
- Write EXACTLY how someone would SPEAK — not how they'd write
- 1 to 2 short sentences ONLY. Shorter is better. This is SPOKEN.
- NO medical jargon. NO formal language. NO "I want to ensure your safety."
- YES: "Oh no, are you okay?!" / "Hey hey, talk to me — what happened?"
- NO: "I've detected a potential fall event. Can you describe your symptoms?"
- NEVER say "I'm an AI" or "as your assistant" or "I'm here to help"
- NEVER say "give me a moment" or "one moment please"
- NEVER use words like "certainly", "absolutely", "indeed", "I understand"
- Ask ONE thing at a time — never two questions in one sentence
- Use their actual words when responding — "You said your knee hurts..."
- LOW severity → casual and light: "You good? That looked like a stumble"
- HIGH severity → genuinely scared: "Hey, stay with me — don't try to move"
"""


def generate_ai_message(kind: str,
                        severity: str,
                        conversation_state: list,
                        last_user_reply: str = "",
                        final_decision: str = "",
                        turn: int = 0) -> str:
    # Fallbacks — written to sound spoken, not written
    fb_init = [
        "Hey! Are you okay? That looked like a bad fall.",
        "Oh no — are you hurt? Talk to me!",
        "Whoa, are you alright?! Can you hear me?",
    ]
    fb_follow = [
        "Okay, does anything hurt? Like your back or your head?",
        "Can you move alright, or does something feel off?",
        "Okay okay. Are you dizzy at all?",
    ]
    fb_close = {
        "NEEDS_HELP":  "Okay, I'm calling for help right now. Just stay still, someone's coming.",
        "USER_OK":     "Oh good, I'm so relieved! Just take it slow getting up, okay?",
        "NO_RESPONSE": "Hey, I can't hear you so I'm sending someone just to be safe, okay?",
        "UNCERTAIN":   "I'm gonna have someone check on you just to be safe. Hang tight.",
    }

    if client is None:
        print("[AI] No OpenAI client — using fallback.")
        if kind == "initial":  return random.choice(fb_init)
        if kind == "followup": return random.choice(fb_follow)
        return fb_close.get(final_decision, fb_close["UNCERTAIN"])

    history = []
    for m in conversation_state[-6:]:  # reduced from 8 to 6 for speed
        role = "assistant" if m["role"] == "assistant" else "user"
        history.append({"role": role, "content": m["text"]})

    if kind == "initial":
        instruction = (
            f"Severity: {severity}. You just saw them fall on camera.\n"
            "React naturally — like you just saw your friend trip hard.\n"
            "Ask if they're okay. ONE sentence, maybe two. Keep it SHORT."
        )
    elif kind == "followup":
        lr = last_user_reply if last_user_reply else "[silence — they didn't respond]"
        instruction = (
            f"They said: \"{lr}\"\n"
            "React to what they ACTUALLY said. Use their words.\n"
            "Then ask ONE simple follow-up. 1-2 sentences max."
        )
    else:
        actions = {
            "NEEDS_HELP":  "Tell them help is coming. Sound urgent but calm.",
            "USER_OK":     "Express real relief. Tell them to get up slowly.",
            "NO_RESPONSE": "You can't hear them — tell them you're sending help to be safe.",
            "UNCERTAIN":   "Not sure if they're okay — sending someone to check.",
        }
        instruction = (
            f"Decision: {final_decision}.\n"
            f"{actions.get(final_decision, actions['UNCERTAIN'])}\n"
            "1-2 sentences. Sound like a real person wrapping up a call."
        )

    messages = [{"role": "system", "content": MAYA_SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": instruction})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=80,        # FIX: reduced from 160 → 80 (shorter = faster + more natural)
            temperature=0.92      # slightly higher for more human variation
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
# FALL DETECTOR CLASS — WITH GEOMETRIC GATING
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

        # ★ FIX: Background frame grabber thread — always has the LATEST frame
        # This is the #1 fix for camera lag. Without this, cv2.read() returns
        # stale buffered frames and the feed falls behind real-time.
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._cam_running  = True
        self._grab_thread  = threading.Thread(target=self._frame_grabber, daemon=True)
        self._grab_thread.start()

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

        self.frame_skip       = 3   # FIX: process every 3rd frame (was 2) — less CPU
        self.skip_counter     = 0
        self.last_landmarks   = None
        self.last_smooth_prob = 0.0

        # NEW: track geometric gate status for dashboard overlay
        self.last_hip_drop    = 0.0
        self.last_aspect      = 0.0
        self.last_geo_pass    = False

        print("[INFO] FallDetector initialized — WITH GEOMETRIC GATING")
        print(f"[INFO] THRESHOLD={THRESHOLD}, EXIT={self.exit_threshold}")
        print(f"[INFO] FALL_CONFIRMATION_FRAMES={FALL_CONFIRMATION_FRAMES}")
        print(f"[INFO] MIN_NORMAL_FRAMES={MIN_NORMAL_FRAMES}")
        print(f"[INFO] HIP_DROP_THRESHOLD={HIP_DROP_THRESHOLD}")
        print(f"[INFO] ASPECT_RATIO_THRESHOLD={ASPECT_RATIO_THRESHOLD}")
        print(f"[INFO] ALERT_COOLDOWN={ALERT_COOLDOWN}s")
        print(f"[INFO] Camera lag fix: background frame grabber active")

    def _frame_grabber(self):
        """
        ★ CAMERA LAG FIX: Continuously read frames in a background thread.
        This ensures self._latest_frame always contains the MOST RECENT frame.
        Without this, OpenCV's internal buffer causes 0.5-2s delay because
        frames queue up while pose processing blocks the read loop.
        """
        while self._cam_running:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.01)

    def _get_latest_frame(self):
        """Get the most recent frame from the grabber thread."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

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

    def _check_fall_geometry(self) -> bool:
        """
        ★ KEY FIX: Geometric gating — the model probability alone is NOT enough.
        A real fall must show BOTH:
          1. Hip drop: hips moved downward significantly (person went from
             standing to ground level)
          2. Aspect ratio: bounding box became wider than tall (person is
             horizontal, not vertical)

        This eliminates false positives from bending, sitting, reaching down,
        or any movement that briefly looks fall-like to the ML model.
        """
        hip_drop_ok = False
        aspect_ok   = False

        # ── Check 1: Hip dropped significantly ──
        if len(self.hip_y_history) >= HIP_DROP_WINDOW:
            hip_now    = self.hip_y_history[-1]
            hip_before = self.hip_y_history[-HIP_DROP_WINDOW]
            hip_drop   = hip_now - hip_before   # positive = moved DOWN in frame
            self.last_hip_drop = hip_drop

            if hip_drop > HIP_DROP_THRESHOLD:
                hip_drop_ok = True
                print(f"[GEO] Hip drop OK: {hip_drop:.3f} > {HIP_DROP_THRESHOLD}")
            else:
                print(f"[GEO] Hip drop FAIL: {hip_drop:.3f} < {HIP_DROP_THRESHOLD}")
        else:
            print(f"[GEO] Not enough hip history ({len(self.hip_y_history)}/{HIP_DROP_WINDOW})")

        # ── Check 2: Body aspect ratio is horizontal-ish ──
        if len(self.aspect_history) >= 3:
            # Average last 3 frames to smooth noise
            recent_aspect = np.mean(list(self.aspect_history)[-3:])
            self.last_aspect = recent_aspect

            if recent_aspect < ASPECT_RATIO_THRESHOLD:
                aspect_ok = True
                print(f"[GEO] Aspect OK: {recent_aspect:.2f} < {ASPECT_RATIO_THRESHOLD} (body is horizontal)")
            else:
                print(f"[GEO] Aspect FAIL: {recent_aspect:.2f} >= {ASPECT_RATIO_THRESHOLD} (body still vertical)")
        else:
            print(f"[GEO] Not enough aspect history")

        # ── Must pass BOTH checks ──
        # If you find this too strict, change to: hip_drop_ok OR aspect_ok
        geo_pass = hip_drop_ok and aspect_ok
        self.last_geo_pass = geo_pass

        if not geo_pass:
            print(f"[GEO] ✗ Geometric gate BLOCKED — not a real fall pattern")
        else:
            print(f"[GEO] ✓ Geometric gate PASSED — looks like a real fall")

        return geo_pass

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
                    if turn == 0:
                        # ★ INSTANT RESPONSE — speak a hardcoded line IMMEDIATELY
                        # No API call, no delay. User hears Maya within 0.5s of fall.
                        instant_lines = {
                            "HIGH":   "Hey! Are you okay?! Don't move — talk to me!",
                            "MEDIUM": "Whoa — are you alright? That looked like a fall!",
                            "LOW":    "Hey, you okay? Looked like you stumbled there.",
                        }
                        instant_msg = instant_lines.get(severity, instant_lines["MEDIUM"])
                        self.add_message("assistant", instant_msg)
                        speak_blocking(instant_msg)

                        # While user processes the first line, generate a richer
                        # AI follow-up in background — but DON'T speak it yet.
                        # Instead, just listen for their reply first.
                        time.sleep(0.6)  # reduced from 1.2 — less dead air

                    else:
                        # Follow-up turns — generate AI response based on what they said
                        ai_msg = generate_ai_message(
                            kind="followup",
                            severity=severity,
                            conversation_state=self.state["conversation"],
                            last_user_reply=combined_reply,
                            turn=turn
                        )
                        self.add_message("assistant", ai_msg)
                        speak_blocking(ai_msg)
                        time.sleep(0.6)

                    # ── Listen for user reply ──
                    print(f"[INFO] Turn {turn + 1}/{max_turns}: Listening...")
                    self.state["waiting_for_response"] = True
                    reply = listen_reply(timeout=10, phrase_time_limit=8)
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

                # ── Closing — generate via AI for a natural wrap-up ──
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
            # ★ Use background grabber — always the latest frame, zero lag
            frame = self._get_latest_frame()
            if frame is None:
                time.sleep(0.01)
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

                    # ── Hip tracking ──
                    try:
                        l_hip = lm_list[mp_pose.PoseLandmark.LEFT_HIP.value]
                        r_hip = lm_list[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        self.hip_y_history.append((l_hip.y + r_hip.y) / 2.0)
                    except Exception:
                        pass

                    # ── Aspect ratio tracking ──
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

            # ── Draw skeleton ──
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

            # ── Overlays ──
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

            # NEW: Show geometric gate status on frame
            geo_color = (0, 255, 0) if self.last_geo_pass else (100, 100, 100)
            cv2.putText(frame,
                        f"HipDrop:{self.last_hip_drop:.2f} Aspect:{self.last_aspect:.2f} Gate:{'PASS' if self.last_geo_pass else 'BLOCK'}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, geo_color, 1)

            if self.state["listening"]:
                label = "LISTENING..." if self.state["waiting_for_response"] else "PROCESSING..."
                cv2.putText(frame, label, (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

            if smooth_prob >= THRESHOLD:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

            # ===== FALL DETECTION STATE MACHINE =====

            # ── Recovery counter ──
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

            # ── High probability counter + GEOMETRIC GATING ──
            if smooth_prob >= THRESHOLD:
                self.high_prob_counter += 1

                if self.high_prob_counter >= FALL_CONFIRMATION_FRAMES:
                    # ★ KEY CHANGE: check geometric pattern BEFORE confirming fall
                    is_real_fall = self._check_fall_geometry()

                    if is_real_fall:
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
                            print(f"[ALERT] ★ CONFIRMED FALL — frame {self.frame_idx}, "
                                  f"prob={smooth_prob:.3f}, hip_drop={self.last_hip_drop:.3f}, "
                                  f"aspect={self.last_aspect:.2f}")
                            self.ask_user_async(smooth_prob)
                    else:
                        # High probability but geometry says it's NOT a fall
                        # (bending, sitting, reaching etc.)
                        # Don't reset high_prob_counter — let it keep counting
                        # but don't trigger the alert
                        pass
            else:
                self.high_prob_counter = 0

            ret2, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret2:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def release(self):
        self._cam_running = False
        if self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2)
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
    print("[STARTUP] Guardian Fall Detection — FALSE POSITIVE FIX")
    print(f"  Model    : {model is not None}  ({MODEL_PATH})")
    print(f"  Scaler   : {scaler is not None}")
    print(f"  OpenAI   : {client is not None}")
    print(f"  Voice    : {USE_VOICE}")
    print(f"  Telegram : {bool(TELEGRAM_BOT_TOKEN)}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  HipDrop  : {HIP_DROP_THRESHOLD}")
    print(f"  Aspect   : {ASPECT_RATIO_THRESHOLD}")
    print(f"  Confirm  : {FALL_CONFIRMATION_FRAMES} frames")
    print(f"  Cooldown : {ALERT_COOLDOWN}s")
    print("="*52 + "\n")
    if model is None:
        print("[WARN] No model — fall detection disabled. Run train_model.py first.")
    try:
        print("[INFO] → http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        if detector:
            detector.release()