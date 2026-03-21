import speech_recognition as sr

print("Available mics:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  [{i}] {name}")

r = sr.Recognizer()
print("\nTesting microphone... say something!")
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)
    print("Listening now (you have 5 seconds)...")
    audio = r.listen(source, timeout=5)
    print("Got audio! Recognizing...")
    text = r.recognize_google(audio)
    print("You said:", text)