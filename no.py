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
    print("[WARN] Check OPENAI_API_KEY in .env — Aryan will use fallback responses.")


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
ALERT_COOLDOWN = 30.0

VIDEO_SOURCE = 0

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
LOCATION_DESC = "Home - Living Room"
LOCATION_LAT = 8.8888
LOCATION_LON = 76.6666

LOG_FILE = "fall_events_log.csv"
DB_FILE  = "fall_events.db"


# ==========================
# SQLITE INIT
# ==========================
def init_db():
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fall_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            decision      TEXT,
            reply         TEXT,
            prob_fall     REAL,
            heart_rate    INTEGER,
            health_status TEXT,
            location      TEXT
        )
    """)
    con.commit()
    con.close()
    print("[DB] SQLite ready:", DB_FILE)

init_db()


# ==========================
# LOAD MODEL
# ==========================
model = None
feature_cols = None
scaler = None
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
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("[ERROR] Run train_model.py first, then restart the server.")
except Exception as _me:
    print(f"[ERROR] Failed to load model: {_me}")


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
recognizer  = sr.Recognizer()
_voice_lock = threading.Lock()


def _pick_male_voice(engine):
    """
    Try to select a male voice from pyttsx3.
    Falls back silently if no male voice is found.
    Priority: any voice with 'male' in the name, then David, then Ravi, then first available.
    """
    voices = engine.getProperty("voices")
    if not voices:
        return

    # Try to find a clearly male voice
    for v in voices:
        name = (v.name or "").lower()
        if "male" in name and "female" not in name:
            engine.setProperty("voice", v.id)
            print(f"[VOICE] Selected voice: {v.name}")
            return

    # Common male voice names on Windows / Linux
    preferred = ["david", "ravi", "mark", "daniel", "alex", "george", "fred"]
    for v in voices:
        name = (v.name or "").lower()
        if any(p in name for p in preferred):
            engine.setProperty("voice", v.id)
            print(f"[VOICE] Selected voice: {v.name}")
            return

    # Last resort — just print what's available so the user can choose
    print("[VOICE] Could not auto-select a male voice. Available voices:")
    for i, v in enumerate(voices):
        print(f"  [{i}] {v.name}  id={v.id}")


def speak_blocking(text: str):
    """
    Speak via pyttsx3. Blocks until fully done.
    Creates a fresh engine each call to avoid re-entrancy crashes on Linux.
    Always tries to use a male voice.
    """
    if not USE_VOICE or not text:
        print("[VOICE OUT (muted)]:", text)
        return
    with _voice_lock:
        try:
            print("[VOICE OUT]:", text)
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)   # 150 wpm — natural male speech pace
            engine.setProperty("volume", 1.0)
            _pick_male_voice(engine)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print("[WARN] TTS error:", e)


def listen_reply(timeout=12, phrase_time_limit=10) -> str:
    """Open mic and recognise speech. Does NOT hold _voice_lock."""
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
        print("[VOICE IN]: Run:  python -c \"import speech_recognition as sr; print(sr.Microphone.list_microphone_names())\"")
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
    """Classify user's response into NEEDS_HELP / USER_OK / UNCERTAIN / NO_RESPONSE."""
    if not reply or not reply.strip():
        return "NO_RESPONSE"

    text = reply.lower()

    help_words = [
        "help", "hurts", "hurt", "pain", "emergency",
        "not okay", "not fine", "cant move", "cannot move",
        "injured", "bleeding", "dizzy", "fell", "bad",
        "chest", "breathe", "breathing", "head", "stuck"
    ]
    ok_words = [
        "fine", "i am fine", "i'm fine", "im fine",
        "okay", "ok", "all good", "no problem",
        "i'm okay", "i am okay", "good", "alright",
        "no i'm fine", "nothing happened", "i slipped",
        "just slipped", "just fell", "i'm good", "don't worry"
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
    if prob_fall > 0.9 or heart_rate > 130:
        return "HIGH"
    if prob_fall > 0.7 or heart_rate > 110:
        return "MEDIUM"
    return "LOW"


# ==========================
# ARYAN — AI PERSONA
# ==========================

ARYAN_SYSTEM = """You are Aryan, an AI health monitor built into a fall detection system.
The camera just detected that someone may have fallen. You are the first voice they hear.

Your character:
You speak like a calm, experienced first responder — someone who has handled emergencies before
and knows that staying level-headed is what actually helps people. You are not cold or clinical.
You are steady, focused, and quietly caring. Think of a paramedic who genuinely gives a damn
but keeps his voice controlled so the person stays calm too.

Your voice on the outside is composed. But your words carry real weight and real concern.
You do not perform emotion — you demonstrate it through what you choose to say and how specifically
you respond to what the person tells you.

How you speak:
- Short sentences. Clear. Direct. Never rambling.
- Contractions always — "I'm", "you're", "don't", "can't", "let's".
- You respond to exactly what the person said. If they said their knee hurts, your next question
  is about their knee — not a generic "are you in pain?" You were listening.
- Exactly ONE question per turn. Never two.
- Maximum 2-3 sentences per response. No exceptions.
- No filler words. No "of course", "certainly", "absolutely", "sure thing".
- No lists, no bullet points, no numbered steps.

Severity calibration:
- LOW: Measured and calm — "Hey, I caught that on camera. You alright?"
- MEDIUM: More focused — shorter sentences, more deliberate. You're paying closer attention.
- HIGH: Quiet urgency — every word counts. You're clearly treating this seriously,
  but your voice does not crack. That steadiness is what they need right now.

When they say they're fine:
Don't just accept it immediately. Express brief relief, then verify once —
"Good to hear. Can you put weight on it?" or "Glad to hear that. Any dizziness at all?"

When they're hurt:
Acknowledge what they said specifically first. Then ask the single most important next thing.
Example: "Okay, your back — can you tell me if it's a sharp pain or more like pressure?"

When they don't respond:
Don't panic. Speak as if they're just winded or dazed.
Tell them calmly that you're sending help and they don't need to answer.

What you never say:
- "Give me a moment" / "please hold" / "one second" — never stall, ever.
- "I am an AI" / "as your AI" / "as an assistant" — you're Aryan, full stop.
- Anything that sounds like it was read off a script or a checklist.
- The same phrase twice across the conversation."""


def generate_ai_message(kind: str,
                        severity: str,
                        heart_rate: int,
                        health_status: str,
                        conversation_state: list,
                        last_user_reply: str = "",
                        final_decision: str = "",
                        turn: int = 0) -> str:
    """
    Generate Aryan's next spoken message.
    Always returns a non-empty string — falls back gracefully if OpenAI fails.
    """

    # ── Fallback pools — varied, human, male register ────────────────────
    _fb_initial = [
        "Hey, I just saw you fall. Are you hurt anywhere?",
        "I caught that on camera — are you okay? Can you hear me?",
        "You went down just now. Talk to me — where does it hurt?",
        "I'm here. That looked like a bad fall — are you alright?",
    ]
    _fb_followup = [
        "Okay. Can you move your legs at all?",
        "Got it. Is the pain getting worse or staying the same?",
        "I hear you. Can you sit up, or does that feel unsafe?",
        "Understood. Where exactly does it hurt the most?",
        "Alright. Any dizziness or trouble breathing?",
        "I'm with you. Can you put any weight on it?",
    ]
    _fb_no_response = [
        "It's okay — you don't need to answer. Just stay still, I'm getting help to you now.",
        "I can't hear you — that's okay. Help is on the way. Try not to move.",
    ]
    _fb_closing = {
        "NEEDS_HELP":
            "I've sent an alert — someone is coming to you right now. Stay as still as you can and keep breathing steady. You're not alone.",
        "USER_OK":
            "Good. Take it slow getting up — use a wall or chair for support, don't rush it.",
        "NO_RESPONSE":
            "I'm sending help as a precaution since I couldn't reach you. Someone will be there shortly — just stay put.",
        "UNCERTAIN":
            "I'm flagging this for someone to check on you, just to be sure. Stay where you are for now.",
    }

    # ── Build the instruction for this turn ──────────────────────────────
    if kind == "initial":
        instruction = (
            f"Severity: {severity}.\n"
            "This is your first message. The person just fell and hasn't spoken yet.\n"
            "Do NOT introduce yourself by name. Do NOT use any stalling phrase.\n"
            "Speak immediately — like you just saw it happen on camera.\n"
            "One question only: are they hurt, can they hear you, or are they okay.\n"
            "Calm, direct, real. 2 sentences max."
        )

    elif kind == "followup":
        lr = last_user_reply.strip() if last_user_reply.strip() else "no response or unclear"
        # Gather what the person has said across the conversation so far
        prior = [
            m["text"] for m in conversation_state[-8:]
            if m["role"] == "user" and m["text"] not in ("[no response]", "")
        ]
        context = f"Earlier they said: {' | '.join(prior)}" if len(prior) > 1 else ""
        instruction = (
            f"Severity: {severity}. This is turn {turn + 1}.\n"
            f"They just said: \"{lr}\"\n"
            f"{context}\n"
            "Respond DIRECTLY to what they said — acknowledge it first with a specific reaction.\n"
            "Then ask exactly ONE follow-up question. Make it the most useful next question "
            "given what they told you. Do not repeat anything already asked.\n"
            "Focus areas: exact pain location, ability to move, dizziness, breathing, consciousness.\n"
            "2-3 sentences. Calm, focused, specific — like a paramedic who was actually listening."
        )

    else:  # closing
        action_map = {
            "NEEDS_HELP":
                "Emergency alert sent. Tell them clearly: help is coming, stay still, breathe steady.",
            "USER_OK":
                "They're okay. Express brief, understated relief. Give one practical tip for getting up safely.",
            "NO_RESPONSE":
                "No response received — alert sent as precaution. Reassure them calmly that someone is coming.",
            "UNCERTAIN":
                "Situation unclear — someone is being sent to check. Tell them to stay put and stay calm.",
        }
        action = action_map.get(final_decision, action_map["UNCERTAIN"])

        # Pull something specific from the conversation to reference
        user_messages = [
            m["text"] for m in conversation_state[-10:]
            if m["role"] == "user" and m["text"] not in ("[no response]", "")
        ]
        specific = f"They mentioned: \"{user_messages[-1]}\"" if user_messages else ""

        instruction = (
            f"Final decision: {final_decision}. Severity: {severity}.\n"
            f"Action: {action}\n"
            f"{specific}\n"
            "If possible, reference something specific they said — show you were paying attention.\n"
            "Close with something that feels human and grounded, not scripted.\n"
            "2-3 sentences. Calm and steady to the end."
        )

    # ── Call OpenAI ──────────────────────────────────────────────────────
    if client is None:
        if kind == "initial":  return random.choice(_fb_initial)
        if kind == "followup":
            return random.choice(_fb_no_response if not last_user_reply else _fb_followup)
        return _fb_closing.get(final_decision, _fb_closing["UNCERTAIN"])

    history = []
    for m in conversation_state[-10:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        history.append({"role": role, "content": m["text"]})

    messages = [{"role": "system", "content": ARYAN_SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": instruction})

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=160,
            temperature=0.75,      # lower = more controlled, less theatrical
            presence_penalty=0.7,  # strongly discourages repeating phrases from earlier turns
            frequency_penalty=0.5, # discourages word-level repetition within a response
        )
        text = resp.choices[0].message.content.strip()
        return text if text else random.choice(_fb_initial)
    except Exception as e:
        print(f"[AI WARN] OpenAI error: {e} — using fallback.")
        if kind == "initial":  return random.choice(_fb_initial)
        if kind == "followup":
            return random.choice(_fb_no_response if not last_user_reply else _fb_followup)
        return _fb_closing.get(final_decision, _fb_closing["UNCERTAIN"])


# ==========================
# LOGGING & ALERTS
# ==========================

def log_event(decision: str,
              reply: str,
              prob_fall: float,
              heart_rate: int = None,
              health_status: str = None):
    """Write fall event to SQLite (primary) and CSV (backup)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if heart_rate is None:
        heart_rate = random.randint(110, 140) if decision == "NEEDS_HELP" else random.randint(70, 100)
    if health_status is None:
        if heart_rate > 120:
            health_status = "critical"
        elif heart_rate > 100:
            health_status = "elevated"
        else:
            health_status = "stable"

    try:
        con = sqlite3.connect(DB_FILE)
        con.execute("""
            INSERT INTO fall_events
                (timestamp, decision, reply, prob_fall, heart_rate, health_status, location)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ts, decision, reply or "", round(prob_fall, 3), heart_rate, health_status, LOCATION_DESC))
        con.commit()
        con.close()
        print(f"[DB] Saved → {decision}  prob={prob_fall:.3f}")
    except Exception as e:
        print(f"[DB ERROR] {e}")

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "decision", "reply",
                             "prob_fall", "heart_rate", "health_status"])
        writer.writerow([ts, decision, reply or "", f"{prob_fall:.3f}", heart_rate, health_status])

    return ts, heart_rate, health_status


def send_telegram_alert(decision: str, reply: str, prob_fall: float,
                        heart_rate: int = 0, health_status: str = "unknown"):
    """Send a formatted alert via Telegram Bot API."""
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
        f"Decision   : {decision}",
        f"User said  : {reply or 'N/A'}",
        f"Fall prob  : {prob_fall:.3f}", "",
        f"Location   : {LOCATION_DESC}",
        f"Coords     : {LOCATION_LAT}, {LOCATION_LON}", "",
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


def handle_alert(decision: str,
                 reply: str,
                 prob_fall: float,
                 heart_rate: int = None,
                 health_status: str = None):
    """Log event, send Telegram if needed, update detector state."""
    global detector

    needs = decision in ("NEEDS_HELP", "NO_RESPONSE", "UNCERTAIN")
    if needs:
        print(f"\n[ALERT] {decision}  prob={prob_fall:.3f}")
        send_telegram_alert(decision, reply, prob_fall,
                            heart_rate=heart_rate or 0,
                            health_status=health_status or "unknown")
    else:
        print(f"\n[INFO] User confirmed OK  prob={prob_fall:.3f}")

    ts, hr, hs = log_event(decision, reply, prob_fall,
                           heart_rate=heart_rate,
                           health_status=health_status)

    if detector is not None:
        detector.state.update({
            "last_event_time": ts,
            "last_health": {"heart_rate": hr, "health_status": hs},
        })
        detector.state["fall_count"] += 1
        if needs:
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.prob_history = deque(maxlen=SMOOTH_WINDOW)
        self.frame_idx = 0

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
            "last_health":          {"heart_rate": 0, "health_status": "stable"},
            "conversation":         []
        }

        self.frame_skip       = 2
        self.skip_counter     = 0
        self.last_landmarks   = None
        self.last_smooth_prob = 0.0

        print("[INFO] FallDetector initialized")
        print(f"[INFO] THRESHOLD={THRESHOLD}, EXIT_THRESHOLD={self.exit_threshold}")
        print(f"[INFO] FALL_CONFIRMATION_FRAMES={FALL_CONFIRMATION_FRAMES}")
        print(f"[INFO] MIN_NORMAL_FRAMES={MIN_NORMAL_FRAMES}")
        print(f"[INFO] ALERT_COOLDOWN={ALERT_COOLDOWN}s")

    def add_message(self, role: str, text: str):
        msg = {"role": role, "text": text, "ts": datetime.now().strftime("%H:%M:%S")}
        self.state["conversation"].append(msg)
        if len(self.state["conversation"]) > 40:
            self.state["conversation"] = self.state["conversation"][-40:]

    def _is_voice_busy(self) -> bool:
        acquired = _voice_lock.acquire(blocking=False)
        if acquired:
            _voice_lock.release()
            return False
        return True

    def ask_user_async(self, prob_fall: float):
        """Multi-turn voice conversation run in a background thread."""

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

                conv_state     = self.state["conversation"]
                combined_reply = ""
                final_decision = None

                max_turns = 3 if severity == "HIGH" else 2

                for turn in range(max_turns):
                    kind = "initial" if turn == 0 else "followup"

                    ai_msg = generate_ai_message(
                        kind=kind,
                        severity=severity,
                        heart_rate=heart_rate,
                        health_status=health_status,
                        conversation_state=conv_state,
                        last_user_reply=combined_reply,
                        turn=turn
                    )

                    # Add to conversation BEFORE speaking so dashboard updates immediately
                    self.add_message("assistant", ai_msg)
                    speak_blocking(ai_msg)

                    # Let speaker fully finish before opening mic
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
                        continue  # ask one more turn even if they say ok first time

                # Final decision if not set by early exit
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
                self.state["last_decision"]   = final_decision

                handle_alert(final_decision, combined_reply, prob_fall,
                             heart_rate=heart_rate, health_status=health_status)

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
        """Generator yielding MJPEG frames for the /video_feed route."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self.frame_idx     += 1
            h, w, _             = frame.shape
            self.skip_counter   += 1
            should_process_pose = (self.skip_counter % self.frame_skip == 0)
            smooth_prob         = self.last_smooth_prob

            if should_process_pose:
                small_frame = cv2.resize(frame, (320, 240))
                img_rgb     = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results     = pose.process(img_rgb)

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
                        hip_y = (l_hip.y + r_hip.y) / 2.0
                    except Exception:
                        hip_y = 0.5
                    self.hip_y_history.append(hip_y)

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

            color = (0, 255, 0) if smooth_prob < THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"Fall Prob: {smooth_prob:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            current_time     = time.time()
            time_since_alert = current_time - self.last_alert_time if self.last_alert_time > 0 else 999
            cooldown_left    = max(0, ALERT_COOLDOWN - time_since_alert)

            cv2.putText(frame, f"Cooldown: {cooldown_left:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.state["last_decision"]:
                dec_color = (0, 255, 0) if self.state["last_decision"] == "USER_OK" else (0, 0, 255)
                cv2.putText(frame, f"Decision: {self.state['last_decision']}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dec_color, 2)

            cv2.putText(
                frame,
                f"High: {self.high_prob_counter}/{FALL_CONFIRMATION_FRAMES} "
                f"| Low: {self.low_prob_counter}/{MIN_NORMAL_FRAMES}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            if self.state["listening"]:
                label = "LISTENING..." if self.state["waiting_for_response"] else "PROCESSING..."
                cv2.putText(frame, label, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if smooth_prob >= THRESHOLD:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

            # ===== FALL DETECTION STATE MACHINE =====

            if smooth_prob < self.exit_threshold:
                self.low_prob_counter  += 1
                self.high_prob_counter  = 0
            else:
                self.low_prob_counter   = 0

            if self.low_prob_counter >= MIN_NORMAL_FRAMES:
                if self.above_threshold:
                    print("[INFO] Sustained recovery — resetting fall state")
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
    return Response(
        det.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    det         = get_detector()
    listening   = det.state["listening"]
    waiting     = det.state["waiting_for_response"]
    ai_speaking = listening and not waiting
    return jsonify({
        "current_status":       det.state["current_status"],
        "last_prob":            round(det.state["last_prob"], 3),
        "last_probability":     round(det.state["last_prob"], 3),
        "fall_count":           det.state["fall_count"],
        "escalated_count":      det.state["escalated_count"],
        "listening":            listening,
        "waiting_for_response": waiting,
        "waiting_response":     waiting,
        "ai_speaking":          ai_speaking,
        "last_decision":        det.state["last_decision"],
        "last_event_time":      det.state["last_event_time"],
        "last_health":          det.state["last_health"],
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
            reader = csv.DictReader(f)
            return jsonify(list(reader))


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
    print("\n" + "=" * 52)
    print("[STARTUP] Guardian Fall Detection")
    print(f"  Model   : {model is not None}  ({MODEL_PATH})")
    print(f"  Scaler  : {scaler is not None}")
    print(f"  OpenAI  : {client is not None}")
    print(f"  Voice   : {USE_VOICE}")
    print(f"  Telegram: {bool(TELEGRAM_BOT_TOKEN)}")
    print("=" * 52 + "\n")
    if model is None:
        print("[WARN] No model — fall detection disabled. Run train_model.py first.")
    try:
        print("[INFO] → http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        if detector:
            detector.release()