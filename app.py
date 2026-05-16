"""
app.py  —  FaceTrack
"""

import os, cv2, time, uuid, logging, threading, numpy as np, queue as _Q # type: ignore
from datetime import datetime, date, timedelta
from flask import Flask, Response, request, jsonify, render_template, redirect, url_for # type: ignore

from ai_assistant    import ai_assistant
from database        import init_db, get_session, Person, AttendanceLog
from face_registry   import (FaceRegistry, DEEPFACE_LOCK, align_face,
                              quality_score, QUALITY_THRESHOLD,
                              identity_cache, extract_faces)
from servo_controller import servo_ctrl
from Speaker          import speaker
from deepface         import DeepFace # type: ignore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "facetrack"

SNAPSHOTS_DIR = os.path.join("static", "snapshots")
PHOTOS_DIR    = os.path.join("static", "photos")
for _d in (SNAPSHOTS_DIR, PHOTOS_DIR):
    os.makedirs(_d, exist_ok=True)

init_db()
registry = FaceRegistry()

# ── Tuning — all overridable via environment variables ────────────────────────
from dotenv import load_dotenv # type: ignore
load_dotenv()

def _ei(key, default): return int(os.getenv(key, default))
def _ef(key, default): return float(os.getenv(key, default))

CAPTURE_FPS          = _ei("CAPTURE_FPS",          30)
DETECTION_INTERVAL   = _ef("DETECTION_INTERVAL",    0.08)  # ~12 fps detection
RECOGNITION_INTERVAL = _ef("RECOGNITION_INTERVAL",  0.20)  # 5 fps recognition
ATTENDANCE_COOLDOWN  = _ei("ATTENDANCE_COOLDOWN",   60)
IOU_MATCH_THRESHOLD  = _ef("IOU_MATCH_THRESHOLD",   0.25)
MAX_TRACK_AGE        = _ei("MAX_TRACK_AGE",         8)
IDENTITY_FLIP_THRESH = _ei("IDENTITY_FLIP_THRESH",  10)
ENROLL_SHOTS         = _ei("ENROLL_SHOTS",          10)
UNKNOWN_SPEAK_CD     = _ef("UNKNOWN_SPEAK_CD",      10.0)
JPEG_QUALITY         = _ei("JPEG_QUALITY",          75)    # 50-85; lower=faster
BBOX_SMOOTH_ALPHA    = _ef("BBOX_SMOOTH_ALPHA",     0.40)  # bbox EMA (0.3-0.6)
SERVO_UPDATE_HZ      = _ef("SERVO_UPDATE_HZ",       30.0)  # max servo cmds/sec

# ── Shared state ───────────────────────────────────────────────────────────────
_raw_frame       = None          # latest BGR from camera  (cam→det)
_raw_frame_lock  = threading.Lock()
_frame_ready     = False

_current_frame   = None          # annotated frame  (cam→stream)
_frame_lock      = threading.Lock()

_det_results     : list  = []    # detection results  (det→cam draw)
_det_lock        = threading.Lock()

_recog_mailbox   : _Q.Queue = _Q.Queue(maxsize=1)
_recog_lock      = threading.Lock()
_recog_map       : dict  = {}
_recog_stop      : threading.Event = threading.Event()

_cam_running     = False
_cam_thread      = None
_det_thread      = None
_detected_cams   : list  = []
_active_cam_idx  = 0

_live_faces      : list  = []
_live_lock       = threading.Lock()

_tracks          : list  = []
_tracks_lock     = threading.Lock()
_next_tid        = 0

_tracking_target : str | None = None
_last_recog_log  : dict  = {}
_bbox_smooth     : dict  = {}   # track_id → smoothed (x1,y1,x2,y2) for display
_last_spoken_unk : float = 0.0
_alerts          : list  = []
_user_cache      : dict  = {}   # employee_id → (user_dict, timestamp) 30s TTL

_cap_mode  = False
_cap_buf   : list = []
_cap_target: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _add_alert(level, message, user_id=None):
    _alerts.append({"id": str(uuid.uuid4()), "level": level,
                    "message": message, "user_id": user_id,
                    "timestamp": datetime.now().isoformat(), "resolved": False})
    if len(_alerts) > 200:
        _alerts.pop(0)


def _iou(a, b):
    ax1,ay1,aw,ah = a; bx1,by1,bw,bh = b
    ax2,ay2 = ax1+aw,ay1+ah; bx2,by2 = bx1+bw,by1+bh
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter == 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def _match_tracks(tracks, detections):
    d2t, used = {}, set()
    for di,db in enumerate(detections):
        best_iou,best_ti = 0.0,-1
        for ti,trk in enumerate(tracks):
            if ti in used: continue
            iou = _iou(db, trk["bbox"])
            if iou > best_iou: best_iou,best_ti = iou,ti
        if best_iou >= IOU_MATCH_THRESHOLD and best_ti >= 0:
            d2t[di] = best_ti; used.add(best_ti)
    return d2t


def _detect_cameras():
    labels = ["Webcam","USB Cam A","USB Cam B"]
    found  = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
        if cap.isOpened():
            found.append({"index":len(found),"device_id":i,
                          "label":labels[len(found)] if len(found)<len(labels) else f"Camera {i}",
                          "active":len(found)==0})
            cap.release()
    return found


def _get_emotion(crop):
    try:
        with DEEPFACE_LOCK:
            r = DeepFace.analyze(crop, actions=["emotion"],
                                 enforce_detection=False, silent=True)
        return r[0]["dominant_emotion"]
    except Exception: return None


def _person_to_user(person):
    session = get_session()
    try:
        logs = (session.query(AttendanceLog).filter_by(person_id=person.id)
                .order_by(AttendanceLog.check_in.desc()).limit(20).all())
        visit_log = [{"timestamp": lg.check_in.isoformat() if lg.check_in else "",
                      "location": "Main Entrance",
                      "duration_sec": int((lg.check_out-lg.check_in).total_seconds())
                                      if lg.check_out and lg.check_in else 0}
                     for lg in logs]
        snaps = [{"filename":fn,"timestamp":fn}
                 for fn in sorted(os.listdir(SNAPSHOTS_DIR))
                 if fn.startswith(person.employee_id)]
        first = logs[-1].check_in.isoformat() if logs else person.created_at.isoformat()
        last  = logs[0].check_in.isoformat()  if logs else person.created_at.isoformat()
        return {"id":person.employee_id,"db_id":person.id,"name":person.name,
                "role":person.role,"department":person.department,
                "status":"flagged" if not person.is_active else "active",
                "first_seen":first,"last_seen":last,
                "visit_count":len(logs),"visit_log":visit_log,
                "snapshots":snaps,"is_known":True}
    finally:
        session.close()


def _get_all_users():
    session = get_session()
    try: return [_person_to_user(p) for p in session.query(Person).all()]
    finally: session.close()


def _get_user(eid):
    session = get_session()
    try:
        p = session.query(Person).filter_by(employee_id=eid).first()
        return _person_to_user(p) if p else None
    finally: session.close()


def _stats():
    session = get_session()
    try:
        return {"total_tracked": session.query(Person).count(),
                "active_today":  session.query(AttendanceLog).filter_by(date=date.today()).count(),
                "flagged":       session.query(Person).filter_by(is_active=False).count(),
                "total_snapshots": len(os.listdir(SNAPSHOTS_DIR))}
    finally: session.close()


def _log_attendance(pid, eid, name, conf, crop, emotion, is_active):
    global _last_spoken_unk
    now = time.time()
    if now - _last_recog_log.get(eid, 0) < ATTENDANCE_COOLDOWN: return
    _last_recog_log[eid] = now
    if not is_active:
        speaker.on_flagged_detected(name)
        _add_alert("critical", f"⚑ Flagged person detected: {name}", eid)
    else:
        speaker.on_recognised(name, conf)
    session = get_session()
    try:
        ex = session.query(AttendanceLog).filter_by(person_id=pid, date=date.today()).first()
        if ex: ex.check_out = datetime.now(); ex.emotion = emotion
        else:
            session.add(AttendanceLog(person_id=pid, date=date.today(),
                                      check_in=datetime.now(), confidence=conf,
                                      emotion=emotion, is_present=True))
        snap = f"{eid}_{int(now)}.jpg"
        sv   = cv2.rotate(crop, cv2.ROTATE_180) if crop is not None and crop.size > 0 else crop
        cv2.imwrite(os.path.join(SNAPSHOTS_DIR, snap), sv)
        session.commit()
        log.info("Attendance: %s %.0f%%", name, conf*100)
    except Exception as e:
        session.rollback(); log.exception("Attendance error: %s", e)
    finally: session.close()


def _finalize_enrolment(crops, target):
    photo_path = None
    if crops:
        photo_path = os.path.join(PHOTOS_DIR, f"{target['employee_id']}_enroll.jpg")
        cv2.imwrite(photo_path, cv2.rotate(crops[0], cv2.ROTATE_180)
                    if crops[0] is not None else crops[0])
    p = registry.enroll_person(name=target["name"], employee_id=target["employee_id"],
                                role=target["role"], department=target["department"],
                                face_crops=crops, photo_path=photo_path)
    if p: _add_alert("info", f"New person enrolled: {p.name}", p.employee_id); speaker.on_enrolled(p.name)
    log.info("Enrolment: %s", p.name if p else "failed")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 3 — Recognition Worker
# ══════════════════════════════════════════════════════════════════════════════

def _recognition_worker(stop_event):
    global _last_spoken_unk
    log.info("Recognition worker started.")
    while not stop_event.is_set():
        try: batch = _recog_mailbox.get(timeout=0.5)
        except _Q.Empty: continue
        if batch is None: break
        identity_cache.cleanup()
        new_map = {}
        for item in batch:
            slot = item["slot"]; embedding = item["embedding"]
            crop = item.get("crop"); track_uid = item.get("track_uid", str(slot))
            cached = identity_cache.get(track_uid)
            if cached is not None:
                cpid, cconf = cached
                person = None
                if cpid is not None:
                    s = get_session()
                    try: person = s.query(Person).filter_by(id=cpid).first()
                    finally: s.close()
                new_map[slot] = {"person":person,"confidence":cconf,"emotion":None,"cached":True}
                continue
            person, confidence = registry.recognize(embedding)
            emotion = None
            if person:
                if crop is not None: emotion = _get_emotion(crop)
                _log_attendance(person.id, person.employee_id, person.name, confidence,
                                crop if crop is not None else np.zeros((112,112,3),np.uint8),
                                emotion, person.is_active)
            else:
                now = time.time()
                if now - _last_spoken_unk > UNKNOWN_SPEAK_CD:
                    _last_spoken_unk = now; speaker.on_unknown_detected()
            identity_cache.set(track_uid, person.id if person else None, confidence)
            new_map[slot] = {"person":person,"confidence":confidence,"emotion":emotion,"cached":False}
        new_map["__face_count__"] = len(batch)
        with _recog_lock: _recog_map.clear(); _recog_map.update(new_map)
    log.info("Recognition worker stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 2 — Detection Worker (InsightFace runs here, never in capture loop)
# ══════════════════════════════════════════════════════════════════════════════

def _detection_worker():
    global _next_tid, _frame_ready
    _last_rs = 0.0
    log.info("Detection worker started.")
    while _cam_running:
        t0 = time.time()
        with _raw_frame_lock: frame = _raw_frame
        if frame is None: time.sleep(0.02); continue

        h, w = frame.shape[:2]

        # ── InsightFace inference (the slow part — isolated here) ─────────────
        face_results = extract_faces(frame)

        cur_bboxes = []
        for fi in face_results:
            x1,y1,x2,y2 = fi["bbox"]
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
            bfw=x2-x1; bfh=y2-y1
            cur_bboxes.append((x1,y1,bfw,bfh) if bfw>0 and bfh>0 else (0,0,0,0))

        with _tracks_lock:
            d2t = _match_tracks(_tracks, cur_bboxes)
            for trk in _tracks: trk["age"] += 1
            for di,ti in d2t.items(): _tracks[ti]["bbox"]=cur_bboxes[di]; _tracks[ti]["age"]=0
            o2n = {}; ni = 0
            for oi,t in enumerate(_tracks):
                if t["age"] <= MAX_TRACK_AGE: o2n[oi]=ni; ni+=1
                else: identity_cache.invalidate(str(t["id"]))
            _tracks[:] = [t for t in _tracks if t["age"] <= MAX_TRACK_AGE]
            d2t = {di:o2n[ti] for di,ti in d2t.items() if ti in o2n}
            for di in [i for i in range(len(face_results)) if i not in d2t]:
                _tracks.append({"id":_next_tid,"bbox":cur_bboxes[di],"person":None,
                                 "conf":0.0,"streak":0,"age":0})
                _next_tid = (_next_tid+1)%100000; d2t[di]=len(_tracks)-1
            enriched = []
            for di,fi in enumerate(face_results):
                ti  = d2t.get(di); trk = _tracks[ti] if ti is not None else None
                enriched.append({**fi,"track_idx":ti,"track_id":trk["id"] if trk else -1,
                                  "bbox_xywh":cur_bboxes[di]})

        # Live capture
        global _cap_mode, _cap_buf
        if face_results and _cap_mode and len(_cap_buf) < ENROLL_SHOTS:
            fi = face_results[0]; x1,y1,x2,y2 = fi["bbox"]
            rc = frame[max(0,y1):y2, max(0,x1):x2]
            if rc.size > 0 and quality_score(rc) >= QUALITY_THRESHOLD:
                _cap_buf.append(frame.copy())
            if len(_cap_buf) >= ENROLL_SHOTS:
                _cap_mode = False
                threading.Thread(target=_finalize_enrolment,
                                 args=(_cap_buf.copy(), dict(_cap_target)), daemon=True).start()
                _cap_buf.clear()

        with _det_lock: _det_results.clear(); _det_results.extend(enriched)

        # Feed recognition mailbox
        now = time.time()
        if face_results and (now - _last_rs) > RECOGNITION_INTERVAL:
            batch = []
            for si,fi in enumerate(enriched):
                emb = fi.get("embedding")
                if emb is None: continue
                x1,y1,x2,y2 = fi["bbox"]
                rc = frame[max(0,y1):y2, max(0,x1):x2].copy() if (y2>y1 and x2>x1) else None
                uid = str(fi["track_id"]) if fi["track_id"] >= 0 else f"unk-{si}"
                batch.append({"slot":si,"embedding":emb,"crop":rc,"track_uid":uid})
            if batch:
                try: _recog_mailbox.get_nowait()
                except _Q.Empty: pass
                try: _recog_mailbox.put_nowait(batch); _last_rs = now
                except _Q.Full: pass
        elif not face_results:
            with _recog_lock: _recog_map.clear()

        elapsed = time.time()-t0
        slp = DETECTION_INTERVAL-elapsed
        if slp > 0: time.sleep(slp)
    log.info("Detection worker stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD 1 — Camera Worker (Capture + Draw loop — NEVER calls InsightFace)
# ══════════════════════════════════════════════════════════════════════════════

def _camera_worker(device_id: int):
    global _raw_frame, _current_frame, _cam_running, _frame_ready, _det_thread

    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        log.error("Cannot open camera device %d", device_id)
        _cam_running = False; return

    for prop,val in [(cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),
                     (cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_AUTOFOCUS,0),
                     (cv2.CAP_PROP_AUTO_EXPOSURE,1),(cv2.CAP_PROP_BUFFERSIZE,1)]:
        cap.set(prop, val)

    _consec_fail = 0; MAX_CF = 90; sdl = time.time()+8.0

    # Launch detection thread now that camera is confirmed open
    _det_thread = threading.Thread(target=_detection_worker, daemon=True, name="DetectionWorker")
    _det_thread.start()
    log.info("Camera worker started → device %d (30 fps capture, ~10 fps detection)", device_id)

    try:
        while _cam_running:
            # ── Grab raw frame as fast as possible ────────────────────────────
            if os.name == "nt":
                ret, frame = cap.read()
            else:
                cap.grab(); ret, frame = cap.retrieve()
                if not ret: ret, frame = cap.read()

            if not ret or frame is None:
                if time.time() < sdl: time.sleep(0.02); continue
                _consec_fail += 1
                if _consec_fail == MAX_CF:
                    log.warning("Camera %d: %d consecutive failures.", device_id, _consec_fail)
                    _add_alert("critical", f"⚠ Camera device {device_id} disconnected", None)
                    blank = np.zeros((480,640,3), np.uint8)
                    cv2.putText(blank,"CAMERA DISCONNECTED",(110,220),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,60,180),2)
                    with _frame_lock: _current_frame = blank
                    with _live_lock:  _live_faces[:] = []
                if _consec_fail >= MAX_CF:
                    cap.release(); time.sleep(3.0)
                    if not _cam_running: break
                    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
                    if cap.isOpened():
                        for p,v in [(cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),
                                    (cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,1)]:
                            cap.set(p,v)
                        _consec_fail = 0
                        _add_alert("info", f"✓ Camera device {device_id} reconnected", None)
                else: time.sleep(0.01)
                continue

            _consec_fail = 0
            frame        = cv2.flip(frame, 1)
            _frame_ready = True

            # ── Publish raw frame for detection thread (single copy) ──────────
            with _raw_frame_lock: _raw_frame = frame.copy()

            # ── Composite latest detection results (non-blocking read) ────────
            draw = frame.copy()
            h, w = draw.shape[:2]

            with _det_lock:   det_snap   = list(_det_results)
            with _recog_lock: recog_snap = dict(_recog_map)
            recog_nf = recog_snap.get("__face_count__", len(recog_snap))
            with _tracks_lock: tracks_snap = [dict(t) for t in _tracks]

            new_faces = []
            for di, fi in enumerate(det_snap):
                x1,y1,x2,y2 = fi["bbox"]
                bx=max(0,x1); by=max(0,y1); bfw=max(0,x2-x1); bfh=max(0,y2-y1)

                # ── Smooth bbox position to eliminate detection-fps jitter ────
                tid = fi.get("track_id", -1)
                raw_box = (float(bx), float(by), float(bx+bfw), float(by+bfh))
                if tid >= 0:
                    prev = _bbox_smooth.get(tid)
                    if prev is None:
                        _bbox_smooth[tid] = raw_box
                    else:
                        α = BBOX_SMOOTH_ALPHA
                        _bbox_smooth[tid] = tuple(α*p + (1-α)*r for p, r in zip(prev, raw_box))
                    sx1, sy1, sx2, sy2 = _bbox_smooth[tid]
                    bx  = int(sx1); by  = int(sy1)
                    bfw = max(0, int(sx2 - sx1)); bfh = max(0, int(sy2 - sy1))
                # Clean up stale track ids
                active_tids = {fi2.get("track_id",-1) for fi2 in det_snap}
                for dead in list(_bbox_smooth.keys()):
                    if dead not in active_tids:
                        del _bbox_smooth[dead]

                cx=bx+bfw//2; cy=by+bfh//2; primary=(di==0)
                ti  = fi.get("track_idx")
                trk = tracks_snap[ti] if (ti is not None and ti < len(tracks_snap)) else None

                # Apply recognition result
                if trk is not None and di in recog_snap and recog_nf == len(det_snap):
                    r = recog_snap[di]
                    if r.get("person"):
                        with _tracks_lock:
                            if ti < len(_tracks):
                                _tracks[ti]["person"]=r["person"]; _tracks[ti]["conf"]=r["confidence"]; _tracks[ti]["streak"]=0
                        trk["person"]=r["person"]; trk["conf"]=r["confidence"]
                    else:
                        with _tracks_lock:
                            if ti < len(_tracks):
                                _tracks[ti]["streak"]=_tracks[ti].get("streak",0)+1
                                if _tracks[ti]["streak"] >= IDENTITY_FLIP_THRESH:
                                    _tracks[ti]["person"]=None; _tracks[ti]["conf"]=0.0

                person=trk["person"] if trk else None
                conf  =trk["conf"]   if trk else 0.0
                emotion=recog_snap.get(di,{}).get("emotion") if recog_nf==len(det_snap) else None

                # Draw box + label
                color=(0,64,255) if person and not person.is_active else \
                      (0,215,255) if person else \
                      (0,120,255) if primary else (60,60,200)
                label=person.name if person else "Unknown"
                ctxt =(f" {conf:.0%}") if person else (f" #{di}" if di>0 else "")
                cv2.rectangle(draw,(bx,by),(bx+bfw,by+bfh),color,2)
                cv2.circle(draw,(cx,cy),4,color,-1)
                tag = label+ctxt
                cv2.rectangle(draw,(bx,by-26),(bx+len(tag)*10+8,by),color,-1)
                cv2.putText(draw,tag,(bx+4,by-8),cv2.FONT_HERSHEY_SIMPLEX,0.52,(10,10,10),1)
                if emotion:
                    cv2.putText(draw,emotion.upper(),(bx,by+bfh+20),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,200,60),1)
                if _cap_mode:
                    pct = int(100*len(_cap_buf)/ENROLL_SHOTS)
                    cv2.putText(draw,f"Capturing… {pct}%",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,128),2)

                # Servo — rate-limited to SERVO_UPDATE_HZ to avoid serial saturation
                uid = person.employee_id if person else ("UNK" if di==0 else f"UNK-{di}")
                _now = time.time()
                _servo_interval = 1.0 / SERVO_UPDATE_HZ
                if not hasattr(_camera_worker, "_last_servo_t"):
                    _camera_worker._last_servo_t = 0.0
                # BUG FIX #6: Always update _last_servo_t when the interval fires,
                # regardless of whether servo_ctrl.update returns True.
                # Previously, if the face wasn't the tracked target (returns False),
                # the timestamp was never updated, so the next time the face *did*
                # become the primary target it would fire a burst of queued-up commands
                # all at once, causing violent servo movement.
                if _now - _camera_worker._last_servo_t >= _servo_interval:
                    _camera_worker._last_servo_t = _now
                    tracked = servo_ctrl.update(cx=cx,cy=cy,frame_w=w,frame_h=h,face_uid=uid,primary=primary)
                else:
                    tracked = False
                if tracked:
                    SC=(0,255,135); arm=16
                    cv2.line(draw,(cx-arm,cy),(cx+arm,cy),SC,2)
                    cv2.line(draw,(cx,cy-arm),(cx,cy+arm),SC,2)
                    cv2.circle(draw,(cx,cy),22,SC,1)
                    cv2.rectangle(draw,(bx-2,by-2),(bx+bfw+2,by+bfh+2),SC,2)
                    cv2.putText(draw,"SERVO",(bx,by+bfh+38),cv2.FONT_HERSHEY_SIMPLEX,0.42,SC,1)

                # Face dict
                snaps = [{"filename":fn,"timestamp":fn}
                         for fn in sorted(os.listdir(SNAPSHOTS_DIR))
                         if fn.startswith(person.employee_id)][-4:] if person else []
                face_dict = {"slot":di,"track_id":trk["id"] if trk else -1,"primary":primary,
                             "user_id":uid,"name":label,"is_known":bool(person),
                             "status":"flagged" if person and not person.is_active else "active" if person else "unknown",
                             "conf":round(conf*100,1),"det_score":round(fi.get("det_score",0)*100,1),
                             "first_seen":"—","last_seen":"—","visit_count":0,"visit_log":[],
                             "snapshots":snaps,"emotion":emotion or "—"}
                if person:
                    # Use cached user info — avoid DB hit on every frame per face
                    _uc = _user_cache.get(person.employee_id)
                    if _uc is None or (time.time() - _uc[1]) > 30.0:
                        _uc_data = _get_user(person.employee_id)
                        _user_cache[person.employee_id] = (_uc_data, time.time())
                        _uc = _user_cache[person.employee_id]
                    u = _uc[0] if _uc else None
                    if u: face_dict.update({"first_seen":u["first_seen"][:10],"last_seen":u["last_seen"][:10],
                                            "visit_count":u["visit_count"],"visit_log":u["visit_log"][:4],
                                            "snapshots":u["snapshots"][-4:]})
                new_faces.append(face_dict)

            with _live_lock: _live_faces[:] = new_faces

            # Status bar
            cv2.rectangle(draw,(0,0),(draw.shape[1],34),(12,18,24),-1)
            ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(draw,f"FaceTrack  |  {ts}  | det ~{int(1/DETECTION_INTERVAL)}fps",
                        (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(80,160,190),1)
            cv2.line(draw,(0,34),(draw.shape[1],34),(0,60,100),1)

            with _frame_lock: _current_frame = draw

    finally:
        _cam_running = False
        try: _recog_mailbox.put_nowait(None)
        except _Q.Full: pass
        cap.release()
        _frame_ready = False
        with _raw_frame_lock:  _raw_frame = None
        with _frame_lock:      _current_frame = None
        with _live_lock:       _live_faces[:] = []
        with _det_lock:        _det_results.clear()
        with _recog_lock:      _recog_map.clear()
        log.info("Camera worker stopped.")


# ── MJPEG stream ───────────────────────────────────────────────────────────────

def _gen_frames():
    while True:
        with _frame_lock: frame = _current_frame
        if frame is None:
            blank = np.zeros((480,640,3), np.uint8)
            cv2.putText(blank,"WAITING FOR CAMERA…",(130,240),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,80,140),2)
            frame = blank
        ret,buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1.0/CAPTURE_FPS)


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Camera
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/start", methods=["POST"])
def camera_start():
    global _cam_running,_cam_thread,_detected_cams,_active_cam_idx,_recog_stop,_tracks,_next_tid
    if _cam_running: return jsonify({"running":True})
    _detected_cams = _detect_cameras()
    if not _detected_cams: return jsonify({"running":False,"error":"No cameras found"})
    _active_cam_idx = 0; _cam_running = True; _recog_stop = threading.Event()
    with _tracks_lock: _tracks.clear()
    _next_tid = 0
    threading.Thread(target=_recognition_worker,args=(_recog_stop,),daemon=True,name="RecognitionWorker").start()
    _cam_thread = threading.Thread(target=_camera_worker,args=(_detected_cams[0]["device_id"],),daemon=True,name="CameraWorker")
    _cam_thread.start()
    return jsonify({"running":True})


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    global _cam_running; _cam_running = False; return jsonify({"running":False})


@app.route("/api/camera/status")
def camera_status(): return jsonify({"running":_cam_running,"frame_ready":_frame_ready})


@app.route("/api/camera/list")
def camera_list():
    return jsonify({"cameras":[{**c,"active":i==_active_cam_idx} for i,c in enumerate(_detected_cams)]})


@app.route("/api/camera/switch", methods=["POST"])
def camera_switch():
    global _cam_running,_cam_thread,_active_cam_idx,_frame_ready
    data=request.json or {}; index=int(data.get("index",0))
    if index>=len(_detected_cams): return jsonify({"ok":False,"error":"Invalid camera index"})
    if index==_active_cam_idx: return jsonify({"ok":True})
    _cam_running=False
    if _cam_thread: _cam_thread.join(timeout=3.0)
    _active_cam_idx=index; _cam_running=True; _frame_ready=False
    _cam_thread=threading.Thread(target=_camera_worker,args=(_detected_cams[index]["device_id"],),daemon=True)
    _cam_thread.start(); return jsonify({"ok":True})


# ══════════════════════════════════════════════════════════════════════════════
# Routes — Faces / Tracking / Enrollment / Persons / Attendance / Alerts / Pages / AI
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/faces")
def get_faces():
    with _live_lock: return jsonify(list(_live_faces))

# Alias used by live.html
@app.route("/api/current_face")
def get_current_face():
    with _live_lock: return jsonify({"faces": list(_live_faces)})

@app.route("/api/track/<user_id>", methods=["POST"])
def track_user(user_id):
    global _tracking_target; _tracking_target=user_id; servo_ctrl.set_target(user_id)
    u=_get_user(user_id); speaker.on_tracking_started(u["name"] if u else user_id)
    return jsonify({"tracking":user_id})

@app.route("/api/track/stop", methods=["POST"])
def track_stop():
    global _tracking_target; _tracking_target=None; servo_ctrl.set_target("STOP")
    speaker.on_tracking_stopped(); return jsonify({"tracking":None})

@app.route("/api/track/status")
def track_status(): return jsonify({"tracking":_tracking_target})

@app.route("/api/track/auto", methods=["POST"])
def track_auto():
    global _tracking_target; _tracking_target="AUTO"; servo_ctrl.set_target("AUTO")
    speaker.on_tracking_started(); return jsonify({"tracking":"AUTO"})

@app.route("/api/persons/enroll", methods=["POST"])
def enroll_person():
    name=request.form.get("name","").strip(); eid=request.form.get("employee_id","").strip()
    role=request.form.get("role","Employee").strip(); dept=request.form.get("department","General").strip()
    if not name or not eid: return jsonify({"success":False,"message":"name and employee_id required"}),400
    imgs=[]
    for f in request.files.getlist("images[]")[:8]:
        img=cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_COLOR)
        if img is not None: imgs.append(img)
    if not imgs: return jsonify({"success":False,"message":"At least one face image required"}),400
    pp=os.path.join(PHOTOS_DIR,f"{eid}_enroll.jpg"); cv2.imwrite(pp,imgs[0])
    p=registry.enroll_person(name,eid,role,dept,imgs,pp)
    if p:
        _add_alert("info",f"Person enrolled: {name}",eid); speaker.on_enrolled(name)
        return jsonify({"success":True,"person":p.to_dict(),"embeddings":len(registry._cache.get(p.id,[]))})
    return jsonify({"success":False,"message":"Enrollment failed — ensure clear face images"}),500

@app.route("/api/capture_enroll", methods=["POST"])
def capture_enroll():
    global _cap_mode,_cap_buf,_cap_target
    data=request.json or {}; name=data.get("name","").strip(); eid=data.get("employee_id","").strip()
    if not name or not eid: return jsonify({"success":False,"message":"name and employee_id required"}),400
    if not _cam_running: return jsonify({"success":False,"message":"Camera not running"}),400
    _cap_target={"name":name,"employee_id":eid,"role":data.get("role","Employee"),"department":data.get("department","General")}
    _cap_buf=[]; _cap_mode=True
    return jsonify({"success":True,"message":f"Capturing {ENROLL_SHOTS} samples for {name}…"})

@app.route("/api/register_unknown", methods=["POST"])
def register_unknown():
    data=request.json or {}; name=data.get("name","").strip(); eid=data.get("employee_id","").strip()
    role=data.get("role","Employee").strip(); dept=data.get("department","General").strip()
    if not name or not eid: return jsonify({"error":"name and employee_id required"}),400
    try:
        with _raw_frame_lock: frame=_raw_frame
        if frame is None: return jsonify({"error":"No active camera frame"}),400
        crops=[frame.copy()]
        faces=extract_faces(frame)
        if faces:
            x1,y1,x2,y2=faces[0]["bbox"]; rc=frame[max(0,y1):y2,max(0,x1):x2]
            if rc.size>0: crops.extend(registry.augment_crops(rc))
        p=registry.enroll_person(name,eid,role,dept,crops)
        if p: _add_alert("info",f"Registered from live: {name}",eid); speaker.on_enrolled(name); return jsonify({"status":"registered","id":eid})
        return jsonify({"error":"Could not extract embedding"}),500
    except Exception as e: log.exception("register_unknown: %s",e); return jsonify({"error":str(e)}),500

@app.route("/api/persons")
def list_persons():
    s=get_session()
    try: return jsonify([p.to_dict() for p in s.query(Person).filter_by(is_active=True).all()])
    finally: s.close()

@app.route("/api/persons/<int:pid>", methods=["DELETE"])
def delete_person(pid): return jsonify({"success":registry.delete_person(pid)})

@app.route("/api/persons/<int:pid>/add_embeddings", methods=["POST"])
def add_person_embeddings(pid):
    try:
        crops=[]
        for f in request.files.getlist("images[]")[:5]:
            img=cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_COLOR)
            if img is not None: crops.append(img)
        if not crops:
            with _raw_frame_lock: frame=_raw_frame
            if frame is None: return jsonify({"success":False,"message":"No images and no active frame"}),400
            crops.append(frame.copy())
        added=registry.add_embeddings(pid,crops)
        return jsonify({"success":added>0,"added":added,"message":f"Added {added} embedding(s)"})
    except Exception as e: return jsonify({"success":False,"message":str(e)}),500

@app.route("/api/persons/embedding_stats")
def embedding_stats(): return jsonify(registry.embedding_stats())

@app.route("/api/attendance")
def get_attendance():
    ds=request.args.get("date"); s=get_session()
    try:
        q=s.query(AttendanceLog)
        if ds:
            try: q=q.filter_by(date=date.fromisoformat(ds))
            except ValueError: return jsonify({"error":"Invalid date"}),400
        return jsonify([l.to_dict() for l in q.order_by(AttendanceLog.check_in.desc()).limit(200).all()])
    finally: s.close()

@app.route("/api/attendance/today")
def today_summary():
    today=date.today(); s=get_session()
    try:
        total=s.query(Person).filter_by(is_active=True).count()
        present=s.query(AttendanceLog).filter_by(date=today,is_present=True).count()
        logs=s.query(AttendanceLog).filter_by(date=today).order_by(AttendanceLog.check_in.desc()).all()
        return jsonify({"date":today.isoformat(),"total":total,"present":present,
                        "absent":max(0,total-present),"logs":[l.to_dict() for l in logs]})
    finally: s.close()

@app.route("/api/alerts")
def get_alerts_api(): return jsonify([a for a in _alerts if not a["resolved"]])

@app.route("/alert/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    for a in _alerts:
        if a["id"]==alert_id: a["resolved"]=True; break
    return redirect(url_for("alerts_page"))

@app.route("/user/<eid>/flag", methods=["POST"])
def flag_user(eid):
    s=get_session()
    try:
        p=s.query(Person).filter_by(employee_id=eid).first()
        if p: p.is_active=False; s.commit(); _add_alert("warning",f"Subject flagged: {p.name}",eid)
    finally: s.close()
    return redirect(url_for("user_detail",employee_id=eid))

@app.route("/user/<eid>/clear", methods=["POST"])
def clear_user(eid):
    s=get_session()
    try:
        p=s.query(Person).filter_by(employee_id=eid).first()
        if p: p.is_active=True; s.commit()
    finally: s.close()
    return redirect(url_for("user_detail",employee_id=eid))

@app.route("/api/servo/configure", methods=["POST"])
@app.route("/api/servo/config", methods=["POST"])
def servo_configure():
    d = request.json or {}
    port = d.get("port") or d.get("pi_url")
    servo_ctrl.configure(port=port, enabled=d.get("enabled"))
    return jsonify(servo_ctrl.status())

@app.route("/api/servo/status")
def servo_status(): return jsonify(servo_ctrl.status())

@app.route("/")
def index(): return render_template("index.html",users=_get_all_users(),stats=_stats())

@app.route("/camera")
def camera_page(): return render_template("camera.html")

@app.route("/live")
def live_page(): return render_template("live.html")

@app.route("/users")
def users_page(): return render_template("users.html",users=_get_all_users(),stats=_stats())

@app.route("/user/<employee_id>")
def user_detail(employee_id):
    user=_get_user(employee_id)
    if user is None: return redirect(url_for("index"))
    return render_template("user_detail.html",user=user)

@app.route("/attendance")
def attendance_page(): return render_template("attendance.html")

@app.route("/alerts")
def alerts_page():
    active=[a for a in _alerts if not a["resolved"]]
    counts={"critical":sum(1 for a in active if a["level"]=="critical"),
            "warning": sum(1 for a in active if a["level"]=="warning"),
            "info":    sum(1 for a in active if a["level"]=="info")}
    return render_template("alerts.html",alerts=active,counts=counts)

def _build_ai_context():
    s=get_session()
    try:
        today=date.today(); persons=s.query(Person).all()
        logs_today=s.query(AttendanceLog).filter_by(date=today).order_by(AttendanceLog.check_in.desc()).all()
        logs_recent=s.query(AttendanceLog).filter(AttendanceLog.date>=today-timedelta(days=2)).order_by(AttendanceLog.check_in.desc()).limit(80).all()
        aa=[a for a in _alerts if not a["resolved"]]
        return {"date":today.isoformat(),"total_enrolled":len(persons),"present_today":len(logs_today),
                "absent_today":max(0,len(persons)-len(logs_today)),
                "flagged_persons":[p.to_dict() for p in persons if not p.is_active],
                "persons":[p.to_dict() for p in persons],"today_logs":[l.to_dict() for l in logs_today],
                "recent_logs":[l.to_dict() for l in logs_recent],"active_alerts":aa,
                "snapshots_count":len(os.listdir(SNAPSHOTS_DIR))}
    finally: s.close()

@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    d=request.json or {}; q=d.get("question","").strip()
    if not q: return jsonify({"error":"question required"}),400
    return jsonify({"answer":ai_assistant.chat(q,_build_ai_context()),"question":q})

@app.route("/api/ai/briefing")
def ai_briefing():
    ctx=_build_ai_context()
    b=ai_assistant.daily_briefing(stats={"total_enrolled":ctx["total_enrolled"],"present_today":ctx["present_today"],
                                          "absent_today":ctx["absent_today"],"flagged_count":len(ctx["flagged_persons"]),
                                          "active_alerts":len(ctx["active_alerts"])},
                                   logs=ctx["today_logs"],alerts=ctx["active_alerts"])
    return jsonify(b)

@app.route("/api/ai/anomalies")
def ai_anomalies():
    ctx=_build_ai_context()
    anomalies=ai_assistant.anomaly_check(recent_logs=ctx["recent_logs"],persons=ctx["persons"],alerts=ctx["active_alerts"])
    for a in anomalies:
        uid=a.get("employee_ids",[None])[0]
        _add_alert(a.get("level","warning"),f"[AI] {a.get('title','')}: {a.get('detail','')}",uid)
    return jsonify({"anomalies":anomalies,"count":len(anomalies)})

@app.route("/api/ai/person/<uid>/summary")
def ai_person_summary(uid):
    user=_get_user(uid)
    if not user: return jsonify({"error":"Person not found"}),404
    return jsonify({"summary":ai_assistant.person_summary(user,user.get("visit_log",[])),"user_id":uid})

if __name__ == "__main__":
    speaker.on_system_ready()
    log.info("FaceTrack v7 → http://localhost:5001  (zero-lag 3-thread pipeline)")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)