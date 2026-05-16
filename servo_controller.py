"""
servo_controller.py
"""

import os
import logging
import threading
import time

from dotenv import load_dotenv
load_dotenv()

from arduino_serial import arduino_bridge

log = logging.getLogger(__name__)


class ServoController:
    """
    Thin wrapper around ArduinoBridge.

    app.py calls:
        servo_ctrl.configure(port=..., enabled=...)
        servo_ctrl.set_target(uid)
        servo_ctrl.update(cx, cy, frame_w, frame_h, face_uid, primary)
        servo_ctrl.status()
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self.enabled  : bool       = False
        self._port    : str        = os.getenv("ARDUINO_PORT", "COM7")
        self._target  : str        = "AUTO"   # track primary face by default

        # Telemetry
        self._last_cx    : int = 0
        self._last_cy    : int = 0
        self._send_count : int = 0
        self._confirmed  : bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def configure(self, port: str | None = None,
                  enabled: bool | None = None,
                  **_kwargs) -> None:
        """
        Update port / enabled state.
        Accepts legacy 'pi_url' kwarg silently via **_kwargs.
        """
        with self._lock:
            changed = False

            if port is not None and port.strip():
                clean = port.strip()
                if clean != self._port:
                    self._port = clean
                    changed    = True

            if enabled is not None and bool(enabled) != self.enabled:
                self.enabled = bool(enabled)
                changed      = True

            if changed:
                if self.enabled:
                    ok = arduino_bridge.start(self._port)
                    if ok:
                        arduino_bridge.centre()
                    else:
                        log.error("Servo: failed to open Arduino on %s.", self._port)
                else:
                    arduino_bridge.stop()

        log.info("Servo configured: enabled=%s  port=%s", self.enabled, self._port)

    def set_target(self, target: str | None) -> None:
        """
        None / "AUTO" → follow primary face (servos active).
        "STOP"        → quench servos.
        employee_id   → lock onto that specific person.
        """
        effective = target if target is not None else "AUTO"
        with self._lock:
            if effective == self._target:
                return
            prev          = self._target
            self._target  = effective
            self._confirmed = False

        arduino_bridge.on_target_acquired()   # reset smoothing ramp

        if effective == "STOP":
            arduino_bridge.quench()
        else:
            arduino_bridge.centre()

        log.info("Servo target: %s", effective)

    def update(self, cx: int, cy: int,
               frame_w: int, frame_h: int,
               face_uid: str, primary: bool) -> bool:
        """
        Called from the camera worker for every detected face (~30 fps).
        Returns True if a move command was dispatched.
        """
        if not self.enabled:
            return False

        with self._lock:
            t = self._target

            # Decide whether this face should be tracked
            if t is None or t == "AUTO":
                if not primary:
                    return False          # only track the largest/first face
            elif t == "STOP":
                return False
            else:
                if t != face_uid:
                    return False          # locked onto a different person

            self._last_cx = int(cx)
            self._last_cy = int(cy)
            self._send_count += 1

        # Pass raw pixel centre — arduino_bridge handles smoothing internally
        arduino_bridge.move(cx, cy)
        return True

    def status(self) -> dict:
        with self._lock:
            bridge = arduino_bridge.status()
            return {
                "enabled":     self.enabled,
                "port":        self._port,
                "target":      self._target,
                "last_cx":     self._last_cx,
                "last_cy":     self._last_cy,
                "send_count":  self._send_count,
                "connected":   bridge["connected"],
                "last_error":  bridge["last_error"],
                "last_pan":    bridge["last_pan"],
                "last_tilt":   bridge["last_tilt"],
            }


# ── Module-level singleton imported by app.py ──────────────────────────────────
servo_ctrl = ServoController()