# backend/speech.py
# Offline TTS (pyttsx3) + real-time STT (VOSK + PyAudio)

import threading
import queue
import json

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
except Exception:
    Model = None
    KaldiRecognizer = None
    pyaudio = None

# ----------------- TEXT TO SPEECH -----------------
_engine = None

def init_tts():
    """Initialize text-to-speech engine."""
    global _engine
    if pyttsx3 is None:
        print("pyttsx3 not available; install it for offline TTS.")
        return
    _engine = pyttsx3.init()
    _engine.setProperty('rate', 170)

def speak(text):
    """Speak text asynchronously (non-blocking)."""
    if _engine is None:
        print("[TTS disabled] would say:", text)
        return

    def _run():
        _engine.say(text)
        _engine.runAndWait()

    threading.Thread(target=_run, daemon=True).start()


# ----------------- SPEECH TO TEXT -----------------
_stt_model = None
_audio_queue = queue.Queue()
_listening = False

# Hardcoded model path
MODEL_PATH = "model/vosk-model-small-en-us-0.15"

def init_stt():
    """Load VOSK STT model from predefined path."""
    global _stt_model
    if Model is None:
        print("VOSK or PyAudio not installed. Install with: pip install vosk pyaudio")
        return
    _stt_model = Model(MODEL_PATH)
    print(f"STT model loaded from {MODEL_PATH}")

def start_listening(callback, device_index=None, samplerate=16000):
    """
    Start real-time microphone listening.
    callback can be:
        - callback(text, final)   -> if you want both partial + final results
        - callback(text)          -> old style, only final results
    """
    global _listening
    if _stt_model is None or pyaudio is None:
        raise RuntimeError("STT not initialized. Call init_stt() first and install vosk/pyaudio.")

    rec = KaldiRecognizer(_stt_model, samplerate)
    rec.SetWords(True)

    pa = pyaudio.PyAudio()
    stream = pa.open(rate=samplerate, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=8000,
                     input_device_index=device_index)
    stream.start_stream()

    _listening = True

    def _listen():
        import json
        while _listening:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text.strip():
                    try:
                        callback(text, True)   # preferred new style
                    except TypeError:
                        callback(text)        # fallback old style
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial.strip():
                    try:
                        callback(partial, False)
                    except TypeError:
                        pass  # ignore if user only expects final results

        stream.stop_stream()
        stream.close()
        pa.terminate()

    threading.Thread(target=_listen, daemon=True).start()
    print("Listening...")
