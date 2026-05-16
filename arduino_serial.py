"""
arduino_serial.py
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv # type: ignore
load_dotenv()

log = logging.getLogger(__name__)

# ── Serial config ─────────────────────────────────────────────────────────────
DEFAULT_BAUD  = 115200
WRITE_TIMEOUT = 0.05
QUEUE_MAXSIZE = 30

# ── Frame dimensions — must match camera resolution in app.py ─────────────────
FRAME_W = 640
FRAME_H = 480

# ── Servo travel limits (must match arduino_servo.ino) ────────────────────────
PAN_MIN,  PAN_MAX  = 40,  140
TILT_MIN, TILT_MAX = 55,  125
PAN_CENTRE  = (PAN_MIN  + PAN_MAX)  // 2    # 90
TILT_CENTRE = (TILT_MIN + TILT_MAX) // 2    # 90

# ── Direction — flip if servo moves wrong way ─────────────────────────────────
# INVERT_PAN  = True  → camera panned right when face is left
# INVERT_TILT = True  → camera tilted down when face is above centre
INVERT_PAN  = False
INVERT_TILT = True    # camera is mounted upside-down on your rig

# ── Tracking params (from working standalone script) ──────────────────────────
INITIAL_DEADZONE  = 30      # px
STABLE_DEADZONE   = 15      # px
INITIAL_SMOOTHING = 0.8     # EMA alpha at acquisition
STABLE_SMOOTHING  = 0.4     # EMA alpha once stable
RAMP_SECONDS      = 2.0     # seconds to ramp initial → stable
MAX_SPEED         = 150     # px/s speed cap


def _px_to_pan(px: float) -> int:
    """X pixel (0..FRAME_W) → pan angle (PAN_MIN..PAN_MAX)."""
    ratio = (px / FRAME_W) * 2.0 - 1.0          # -1..+1, left→right
    if INVERT_PAN:
        ratio = -ratio
    angle = PAN_CENTRE + ratio * (PAN_MAX - PAN_MIN) / 2.0
    return max(PAN_MIN, min(PAN_MAX, round(angle)))


def _px_to_tilt(py: float) -> int:
    """Y pixel (0..FRAME_H) → tilt angle (TILT_MIN..TILT_MAX)."""
    ratio = (py / FRAME_H) * 2.0 - 1.0          # -1..+1, top→bottom
    if INVERT_TILT:
        ratio = -ratio
    angle = TILT_CENTRE + ratio * (TILT_MAX - TILT_MIN) / 2.0
    return max(TILT_MIN, min(TILT_MAX, round(angle)))


@dataclass
class _Cmd:
    kind: str
    pan:  int = PAN_CENTRE
    tilt: int = TILT_CENTRE


class ArduinoBridge:

    def __init__(self):
        self._port   : str | None  = None
        self._baud   : int         = DEFAULT_BAUD
        self._serial               = None
        self._queue  : queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread : threading.Thread | None = None
        self._running: bool        = False
        self._lock   = threading.Lock()

        # Pixel-space smoothing state
        self._smoothed_x  : float = FRAME_W / 2.0
        self._smoothed_y  : float = FRAME_H / 2.0
        self._last_sent_x : float = FRAME_W / 2.0
        self._last_sent_y : float = FRAME_H / 2.0
        self._confirm_t   : float = 0.0
        self._confirmed   : bool  = False

        # Telemetry
        self._connected  : bool       = False
        self._last_error : str | None = None
        self._send_count : int        = 0
        self._last_pan   : int        = PAN_CENTRE
        self._last_tilt  : int        = TILT_CENTRE

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self, port: str | None = None, baud: int = DEFAULT_BAUD) -> bool:
        if self._running:
            self.stop()

        self._port = port or os.getenv("ARDUINO_PORT", "COM7")
        self._baud = baud

        try:
            import serial as _serial
            ser = _serial.Serial(self._port, self._baud,
                                 timeout=1.0, write_timeout=WRITE_TIMEOUT)
            time.sleep(2.0)           # wait for Arduino reboot after DTR
            ser.reset_input_buffer()
            with self._lock:
                self._serial    = ser
                self._connected = True
            log.info("Arduino bridge: connected on %s @ %d baud.", self._port, self._baud)
        except Exception as exc:
            self._last_error = str(exc)
            log.error("Arduino bridge: failed to open %s — %s", self._port, exc)
            return False

        self._reset_smoothing()
        self._running = True
        self._thread  = threading.Thread(target=self._worker,
                                         name="ArduinoBridge", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        try:
            self._queue.put_nowait(_Cmd("stop"))
        except queue.Full:
            pass
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except Exception:
                    pass
        log.info("Arduino bridge stopped.")

    def on_target_acquired(self) -> None:
        """Reset smoothing ramp when a new face is locked."""
        with self._lock:
            self._reset_smoothing()

    def move(self, cx: int, cy: int) -> None:
        """
        Receive raw face-centre pixel from camera worker.
        Applies smoothing + deadzone + speed-cap in pixel space,
        then converts to servo angles and queues the command.
        """
        with self._lock:
            if not self._confirmed:
                self._confirmed   = True
                self._confirm_t   = time.time()
                self._smoothed_x  = float(cx)
                self._smoothed_y  = float(cy)
                self._last_sent_x = float(cx)
                self._last_sent_y = float(cy)
                pan  = _px_to_pan(cx)
                tilt = _px_to_tilt(cy)
                self._enqueue_unsafe(_Cmd("move", pan, tilt))
                return

            now      = time.time()
            t        = min(1.0, (now - self._confirm_t) / RAMP_SECONDS)

            # Adaptive EMA
            alpha            = INITIAL_SMOOTHING - (INITIAL_SMOOTHING - STABLE_SMOOTHING) * t
            self._smoothed_x = alpha * self._smoothed_x + (1.0 - alpha) * cx
            self._smoothed_y = alpha * self._smoothed_y + (1.0 - alpha) * cy

            # Adaptive deadzone
            deadzone = INITIAL_DEADZONE - (INITIAL_DEADZONE - STABLE_DEADZONE) * t
            dx = self._smoothed_x - self._last_sent_x
            dy = self._smoothed_y - self._last_sent_y

            if abs(dx) <= deadzone and abs(dy) <= deadzone:
                return

            # Speed cap
            dist = math.sqrt(dx * dx + dy * dy)
            max_dist = MAX_SPEED / 30.0
            if dist > max_dist:
                scale = max_dist / dist
                dx   *= scale
                dy   *= scale

            new_x = max(0.0, min(float(FRAME_W), self._last_sent_x + dx))
            new_y = max(0.0, min(float(FRAME_H), self._last_sent_y + dy))

            self._last_sent_x = new_x
            self._last_sent_y = new_y

            pan  = _px_to_pan(new_x)
            tilt = _px_to_tilt(new_y)
            self._last_pan  = pan
            self._last_tilt = tilt
            self._enqueue_unsafe(_Cmd("move", pan, tilt))

    def centre(self) -> None:
        with self._lock:
            self._reset_smoothing()
            self._drain_unsafe()
        try:
            self._queue.put_nowait(_Cmd("centre"))
        except queue.Full:
            pass

    def quench(self) -> None:
        self._drain()
        try:
            self._queue.put_nowait(_Cmd("quench"))
        except queue.Full:
            pass

    def status(self) -> dict:
        with self._lock:
            return {
                "connected":   self._connected,
                "port":        self._port,
                "baud":        self._baud,
                "last_pan":    self._last_pan,
                "last_tilt":   self._last_tilt,
                "send_count":  self._send_count,
                "last_error":  self._last_error,
                "queue_depth": self._queue.qsize(),
            }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _reset_smoothing(self) -> None:
        self._smoothed_x  = FRAME_W / 2.0
        self._smoothed_y  = FRAME_H / 2.0
        self._last_sent_x = FRAME_W / 2.0
        self._last_sent_y = FRAME_H / 2.0
        self._confirmed   = False
        self._confirm_t   = 0.0

    def _enqueue_unsafe(self, cmd: _Cmd) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(cmd)
        except queue.Full:
            pass

    def _drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _drain_unsafe(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _send_line(self, line: str) -> bool:
        with self._lock:
            ser = self._serial
        if ser is None or not ser.is_open:
            return False
        try:
            ser.write((line + "\n").encode("ascii"))
            ser.flush()
            return True
        except Exception as exc:
            with self._lock:
                self._connected  = False
                self._last_error = str(exc)
            log.warning("Arduino write error: %s", exc)
            return False

    def _try_reconnect(self) -> None:
        log.info("Arduino bridge: attempting reconnect on %s…", self._port)
        try:
            import serial as _serial
            with self._lock:
                if self._serial:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
            ser = _serial.Serial(self._port, self._baud,
                                 timeout=1.0, write_timeout=WRITE_TIMEOUT)
            time.sleep(2.0)
            ser.reset_input_buffer()
            with self._lock:
                self._serial     = ser
                self._connected  = True
                self._last_error = None
            log.info("Arduino bridge: reconnected.")
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            log.warning("Arduino bridge: reconnect failed — %s", exc)

    def _worker(self) -> None:
        log.info("Arduino bridge worker running.")
        while self._running:
            try:
                cmd = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if cmd.kind == "stop":
                self._send_line("quench")
                break
            elif cmd.kind == "quench":
                ok = self._send_line("quench")
            elif cmd.kind == "centre":
                ok = self._send_line("centre")
            elif cmd.kind == "move":
                ok = self._send_line(f"{cmd.pan},{cmd.tilt}")
            else:
                continue

            if ok:
                with self._lock:
                    self._send_count += 1
                    self._connected   = True
                log.debug("Arduino ← pan=%d tilt=%d", cmd.pan, cmd.tilt)
            else:
                time.sleep(0.5)
                if self._running:
                    self._try_reconnect()

        log.info("Arduino bridge worker stopped.")


# ── Module-level singleton ─────────────────────────────────────────────────────
arduino_bridge = ArduinoBridge()