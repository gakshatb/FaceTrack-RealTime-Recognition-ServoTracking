#include <Servo.h>

// ── Pin assignments ───────────────────────────────────────────────────────────
const int PAN_PIN  = 9;
const int TILT_PIN = 10;

// ── Travel limits (degrees) ───────────────────────────────────────────────────
const int PAN_MIN  = 20,  PAN_MAX  = 160;
const int TILT_MIN = 35,  TILT_MAX = 155;
const int PAN_CENTRE  = (PAN_MIN  + PAN_MAX)  / 2;   // 90
const int TILT_CENTRE = (TILT_MIN + TILT_MAX) / 2;   // 90

// ── BUG FIX #1: Removed Arduino-side EMA smoothing ───────────────────────────
// arduino_serial.py already sends fully-smoothed angles. Applying a second
// EMA here compounds latency — the servo always chases a ghost position that
// is 1-2 full EMA time-constants behind the face. All smoothing now lives
// exclusively in arduino_serial.py where it can be tuned in one place.

// Minimum angle change (degrees) before writing to servo — prevents servo
// buzz/chatter from floating-point truncation noise.
const float MIN_DELTA = 1.0;   // raised from 0.8 — SG90 can't resolve < 1°

// ── Serial ────────────────────────────────────────────────────────────────────
const long BAUD     = 115200;
const int  BUF_SIZE = 32;

// ── State ─────────────────────────────────────────────────────────────────────
Servo panServo;
Servo tiltServo;

char  rxBuf[BUF_SIZE];
int   rxIdx   = 0;
bool  attached = false;

// Current written angles (no EMA — just last commanded value)
float curPan  = PAN_CENTRE;
float curTilt = TILT_CENTRE;

// ── Helpers ───────────────────────────────────────────────────────────────────

int clamp(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void attachServos() {
  if (!attached) {
    panServo.attach(PAN_PIN);
    tiltServo.attach(TILT_PIN);
    attached = true;
  }
}

void detachServos() {
  if (attached) {
    panServo.detach();
    tiltServo.detach();
    attached = false;
  }
}

// Write target angles directly — no local EMA (Python handles smoothing)
void writeServos(float tgtPan, float tgtTilt) {
  if (!attached) return;

  float dp = abs(tgtPan  - curPan);
  float dt = abs(tgtTilt - curTilt);

  // Only write if movement exceeds minimum threshold (avoids servo buzz)
  if (dp > MIN_DELTA || dt > MIN_DELTA) {
    curPan  = tgtPan;
    curTilt = tgtTilt;
    panServo.write((int)curPan);
    tiltServo.write((int)curTilt);
  }
}

// ── Process one complete line from PC ─────────────────────────────────────────

void processLine(char* line) {
  // Trim \r (Windows line endings)
  int len = strlen(line);
  if (len > 0 && line[len - 1] == '\r') line[--len] = '\0';
  if (len == 0) return;

  if (strcmp(line, "centre") == 0) {
    attachServos();
    // Snap immediately — no smoothing delay
    curPan  = PAN_CENTRE;
    curTilt = TILT_CENTRE;
    panServo.write(PAN_CENTRE);
    tiltServo.write(TILT_CENTRE);
    return;
  }

  if (strcmp(line, "quench") == 0) {
    detachServos();
    return;
  }

  // Expect "pan,tilt"
  char* comma = strchr(line, ',');
  if (comma == nullptr) return;   // silently ignore bad format
  *comma = '\0';

  int pan  = clamp(atoi(line),      PAN_MIN,  PAN_MAX);
  int tilt = clamp(atoi(comma + 1), TILT_MIN, TILT_MAX);

  attachServos();
  writeServos((float)pan, (float)tilt);
}

// ── setup / loop ──────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(BAUD);
  while (!Serial) { ; }
  attachServos();
  panServo.write(PAN_CENTRE);
  tiltServo.write(TILT_CENTRE);
  curPan  = PAN_CENTRE;
  curTilt = TILT_CENTRE;
  // Send ready signal once only
  Serial.println("FaceTrack Arduino ready");
}

void loop() {
  // ── BUG FIX #2: Read ALL available serial bytes before any delay ──────────
  // Original code had delay(10) at the bottom, meaning serial data queued up
  // for 10ms between reads. At 115200 baud with 30fps commands this caused
  // commands to accumulate and be processed in bursts, creating servo jitter.
  // Now we drain the entire serial buffer on every loop iteration FIRST,
  // then apply a short yield delay (1ms) only to prevent PWM signal corruption.
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rxBuf[rxIdx] = '\0';
      processLine(rxBuf);
      rxIdx = 0;
    } else if (rxIdx < BUF_SIZE - 1) {
      rxBuf[rxIdx++] = c;
    }
    // If buffer overflow (malformed line), silently reset
    else {
      rxIdx = 0;
    }
  }

  // 1ms yield — just enough to avoid corrupting servo PWM signal timing.
  // 1000 Hz update rate is far more than enough for SG90 (50Hz PWM).
  delay(1);
}
