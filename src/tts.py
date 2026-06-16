import queue
import subprocess
import sys
import threading

_queue = queue.Queue()
_started = False
_lock = threading.Lock()


def _system_say(text):
    if sys.platform == "darwin":
        subprocess.run(["say", text], check=False)


def _worker():
    engine = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
    except Exception as e:
        print(f"[tts] pyttsx3 unavailable ({e}); using system voice fallback")

    while True:
        text = _queue.get()
        if text is None:
            break
        try:
            if engine is not None:
                engine.say(text)
                engine.runAndWait()
            else:
                _system_say(text)
        except Exception as e:
            print(f"[tts] pyttsx3 failed ({e}); using system voice fallback")
            engine = None
            _system_say(text)


def speak(text):
    """Queue text for speech. Non-blocking; safe to call from any thread."""
    global _started
    with _lock:
        if not _started:
            threading.Thread(target=_worker, daemon=True).start()
            _started = True
    _queue.put(text)


if __name__ == "__main__":
    import time

    speak("Hello, this is a test")
    time.sleep(4)
