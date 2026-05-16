"""
Speaker.py
"""

from __future__ import annotations

import heapq
import io
import logging
import os
import tempfile
import threading
import time

log = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────
DEDUP_WINDOW   = 5.0      # seconds before the same phrase can repeat
MAX_QUEUE      = 20       # drop oldest low-priority items if queue grows large
TTS_LANG       = "en"     # gTTS language code

# Priority levels (lower number = higher priority)
PRIORITY_HIGH   = 0
PRIORITY_NORMAL = 1
PRIORITY_LOW    = 2


# ── Backend detection (done once at import time) ───────────────────────────────

_BACKEND = "stdout"   # will be updated below

def _try_init_pygame() -> bool:
    try:
        import pygame
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        log.info("Speaker: pygame mixer initialised.")
        return True
    except Exception as e:
        log.warning("Speaker: pygame init failed (%s).", e)
        return False

def _try_import_gtts() -> bool:
    try:
        from gtts import gTTS  # noqa: F401
        return True
    except ImportError:
        log.warning("Speaker: gTTS not installed.")
        return False

_PYGAME_OK = _try_init_pygame()
_GTTS_OK   = _try_import_gtts()

if _GTTS_OK and _PYGAME_OK:
    _BACKEND = "gtts"
    log.info("Speaker backend: gTTS + pygame.")
else:
    # Try pyttsx3 as fallback
    try:
        import pyttsx3 as _pyttsx3  # noqa: F401
        _BACKEND = "pyttsx3"
        log.info("Speaker backend: pyttsx3 (fallback).")
    except ImportError:
        _BACKEND = "stdout"
        log.warning("Speaker backend: stdout-only (install gTTS + pygame for audio).")


class Speaker:
    """
    Thread-safe, non-blocking text-to-speech service.
    """

    def __init__(self):
        self._running = True
        self._heap : list[tuple] = []
        self._seq  : int         = 0
        self._lock  = threading.Lock()
        self._ready = threading.Event()
        self._dedup: dict[str, float] = {}

        # pyttsx3 engine (only used in pyttsx3 backend)
        self._engine = None

        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="SpeakerWorker")
        self._thread.start()
        log.info("Speaker initialised (backend=%s).", _BACKEND)

    # ── Public API ─────────────────────────────────────────────────────────────

    def say(self, text: str, priority: int = PRIORITY_NORMAL,
            dedup: bool = True) -> None:
        if not text:
            return
        if dedup and self._is_duplicate(text):
            return
        self._enqueue(priority, text)

    def alert(self, text: str) -> None:
        self.say(text, priority=PRIORITY_HIGH, dedup=False)

    def status(self, text: str) -> None:
        self.say(text, priority=PRIORITY_LOW)

    # ── Named event helpers ────────────────────────────────────────────────────

    def on_recognised(self, name: str, confidence: float | None = None) -> None:
        self.say(f"Welcome, {name}", priority=PRIORITY_NORMAL)

    def on_unknown_detected(self) -> None:
        self.say("Unknown person detected", priority=PRIORITY_NORMAL)

    def on_enrolled(self, name: str) -> None:
        self.say(f"{name} has been added to the database successfully",
                 priority=PRIORITY_NORMAL)

    def on_tracking_started(self, name: str | None = None) -> None:
        if name:
            self.status(f"Tracking {name}")
        else:
            self.status("Tracking started")

    def on_tracking_stopped(self) -> None:
        self.status("Tracking stopped")

    def on_flagged_detected(self, name: str) -> None:
        self.alert(f"Warning! Flagged person detected: {name}")

    def on_system_ready(self) -> None:
        self.status("FaceTrack system online")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _is_duplicate(self, text: str) -> bool:
        now = time.time()
        key = text.lower().strip()
        last = self._dedup.get(key, 0.0)
        if now - last < DEDUP_WINDOW:
            return True
        self._dedup[key] = now
        if len(self._dedup) > 200:
            cutoff = now - DEDUP_WINDOW * 4
            self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}
        return False

    def _enqueue(self, priority: int, text: str) -> None:
        with self._lock:
            if len(self._heap) >= MAX_QUEUE:
                self._heap.sort()
                self._heap.pop()
            heapq.heappush(self._heap, (priority, self._seq, text))
            self._seq += 1
        self._ready.set()

    def _dequeue(self) -> str | None:
        with self._lock:
            if self._heap:
                _, _, text = heapq.heappop(self._heap)
                return text
        return None

    def _worker(self):
        if _BACKEND == "pyttsx3":
            self._engine = self._init_pyttsx3()

        log.info("Speaker worker thread running (backend=%s).", _BACKEND)

        while self._running:
            self._ready.wait(timeout=0.5)
            self._ready.clear()
            while True:
                text = self._dequeue()
                if text is None:
                    break
                self._speak_now(text)

        log.info("Speaker worker thread stopped.")

    def _speak_now(self, text: str) -> None:
        print(f"[Speaker] {text}")

        if _BACKEND == "gtts":
            self._speak_gtts(text)
        elif _BACKEND == "pyttsx3" and self._engine:
            self._speak_pyttsx3(text)
        # else: stdout only (already printed above)

    def _speak_gtts(self, text: str) -> None:
        """Synthesise with Google TTS and play via pygame."""
        try:
            from gtts import gTTS
            import pygame

            # Generate MP3 into a BytesIO buffer (no temp file needed)
            buf = io.BytesIO()
            tts = gTTS(text=text, lang=TTS_LANG, slow=False)
            tts.write_to_fp(buf)
            buf.seek(0)

            # Play through pygame
            pygame.mixer.music.load(buf, "mp3")
            pygame.mixer.music.play()

            # Wait for playback to finish (non-blocking poll)
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

        except Exception as e:
            log.warning("gTTS/pygame speak error: %s", e)
            # Try reinitialising mixer
            try:
                import pygame
                pygame.mixer.quit()
                pygame.mixer.init()
            except Exception:
                pass

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            log.warning("pyttsx3 speak error (%s) — reinitialising.", e)
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = self._init_pyttsx3()

    def _init_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   160)
            engine.setProperty("volume", 1.0)
            voices = engine.getProperty("voices")
            if voices:
                en_voices = [v for v in voices
                             if "en" in (v.languages[0].decode()
                                         if v.languages else "").lower()]
                engine.setProperty("voice",
                                   (en_voices[0] if en_voices else voices[0]).id)
            return engine
        except Exception as e:
            log.warning("pyttsx3 init failed: %s", e)
            return None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False
        self._ready.set()
        if _BACKEND == "pyttsx3" and self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        if _BACKEND == "gtts":
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._thread.join(timeout=3.0)
        log.info("Speaker stopped.")


# ── Module-level singleton ─────────────────────────────────────────────────────
speaker = Speaker()
