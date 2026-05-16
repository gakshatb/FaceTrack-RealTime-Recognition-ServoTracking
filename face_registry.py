"""
face_registry.py  —  FaceTrack
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from typing import Optional

import cv2
import numpy as np

from database import FaceEmbedding, Person, get_session

log = logging.getLogger(__name__)

# ── InsightFace initialisation ─────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHT_OK = True
    log.info("insightface library found.")
except ImportError:
    _FaceAnalysis = None
    _INSIGHT_OK   = False
    log.error(
        "insightface not installed! "
        "Run: pip install insightface onnxruntime"
    )

# Backward-compat: DeepFace still used for emotion detection in app.py
from threading import Lock as _Lock
DEEPFACE_LOCK = _Lock()


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# ArcFace embeddings are L2-normalised, so cosine distance = 1 - dot_product
# 0.40 distance ≈ 80% cosine similarity — well-calibrated for ArcFace
TOLERANCE        = 0.40

# RetinaFace minimum detection confidence
DET_CONFIDENCE   = 0.75

# Minimum pixel size for a face crop to be processed
MIN_CROP_PX      = 48

# Laplacian variance quality threshold
QUALITY_THRESHOLD = 0.10

# Maximum distinct embeddings to store per person
MAX_EMBEDDINGS   = 15

# Number of best crops used during enrollment
ENROLL_SHOTS     = 8

# Cosine similarity above which two vectors are considered duplicates
DEDUP_SIMILARITY = 1.0 - 0.12   # 0.88

# ArcFace canonical crop size (must match training)
TARGET_SIZE      = 112

# Model name tag stored in DB
MODEL_NAME       = "insightface_arcface_w600k_r50"

# Identity cache
CONF_SKIP_THRESHOLD = 0.78   # cosine similarity (high = very confident)
IDENTITY_CACHE_TTL  = 5.0
VOTE_WINDOW         = 6

# Directory for per-person face images
FACE_DB_DIR = os.path.join("static", "face_db")


# ══════════════════════════════════════════════════════════════════════════════
# InsightFace app singleton (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════

_insight_app  = None
_insight_lock = threading.Lock()


def _get_insight_app():
    """Lazy-initialise InsightFace FaceAnalysis (thread-safe)."""
    global _insight_app
    if _insight_app is not None:
        return _insight_app
    if not _INSIGHT_OK:
        return None
    with _insight_lock:
        if _insight_app is not None:
            return _insight_app
        try:
            # buffalo_l: full accuracy (RetinaFace det_10g + ArcFace w600k_r50)
            # buffalo_sc: faster/smaller — swap name if CPU is too slow
            app = _FaceAnalysis(
                name      = "buffalo_l",
                root      = "./models",   # weights downloaded/stored here
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            # det_size: detection model input resolution.
            # (320, 320) runs ~3x faster than (640, 480) on CPU with
            # minimal accuracy drop for normal webcam distances.
            # Increase to (640, 480) only if you have a GPU or need
            # to detect small/distant faces.
            app.prepare(ctx_id=0, det_size=(320, 320))
            _insight_app = app
            log.info("InsightFace FaceAnalysis ready (buffalo_l).")
        except Exception as exc:
            log.exception("InsightFace init failed: %s", exc)
            return None
    return _insight_app


# ══════════════════════════════════════════════════════════════════════════════
# Core: detect + embed in one call
# ══════════════════════════════════════════════════════════════════════════════

def extract_faces(frame_bgr: np.ndarray) -> list[dict]:
    """
    Run RetinaFace detection + ArcFace embedding on a full BGR frame.

    Returns a list of dicts (one per detected face), sorted by bounding-box
    area descending (largest face first):
        {
            "bbox":      [x1, y1, x2, y2]  (int pixel coords),
            "kps":       np.ndarray (5×2 landmarks),
            "embedding": np.ndarray (512-d L2-normalised float32),
            "det_score": float (RetinaFace confidence 0–1),
        }

    Returns [] if no faces found or InsightFace not initialised.
    """
    app = _get_insight_app()
    if app is None or frame_bgr is None or frame_bgr.size == 0:
        return []

    try:
        with _insight_lock:
            faces = app.get(frame_bgr)
    except Exception as exc:
        log.warning("extract_faces error: %s", exc)
        return []

    results = []
    for face in faces:
        if float(face.det_score) < DET_CONFIDENCE:
            continue
        bb   = face.bbox.astype(int)  # [x1, y1, x2, y2]
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        emb  = face.normed_embedding  # already L2-normalised float32
        results.append({
            "bbox":      bb,
            "kps":       face.kps,
            "embedding": emb.astype(np.float32),
            "det_score": float(face.det_score),
            "area":      area,
        })

    # Largest face first (primary = index 0)
    results.sort(key=lambda x: -x["area"])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Compatibility shim: align_face()
# ──────────────────────────────────────────────────────────────────────────────
# app.py still calls align_face() in the live-capture enrollment path.
# We keep the signature but implement it via InsightFace so the crop is
# always ArcFace-normalised (112×112).
# ══════════════════════════════════════════════════════════════════════════════

def align_face(frame_bgr: np.ndarray,
               detection,
               frame_h: int, frame_w: int) -> Optional[np.ndarray]:
    """
    Backward-compatible shim for the MediaPipe-era align_face() API.
    Extracts the face crop from the frame using the detection's bounding box.
    Returns a 112×112 BGR crop, or None if crop is too small.

    NOTE: This is used only for live-capture enrollment quality display and
    the enroll-from-frame route. Full recognition uses extract_faces() instead.
    """
    try:
        # detection may be a MediaPipe detection object (legacy) or a dict
        if hasattr(detection, "location_data"):
            # MediaPipe detection object
            bb  = detection.location_data.relative_bounding_box
            x   = max(0, int(bb.xmin   * frame_w))
            y   = max(0, int(bb.ymin   * frame_h))
            w   = min(frame_w - x, int(bb.width  * frame_w))
            h_b = min(frame_h - y, int(bb.height * frame_h))
        elif isinstance(detection, dict) and "bbox" in detection:
            # InsightFace detection dict
            x1, y1, x2, y2 = detection["bbox"]
            x, y = max(0, x1), max(0, y1)
            w    = min(frame_w - x, x2 - x1)
            h_b  = min(frame_h - y, y2 - y1)
        else:
            return None

        if w < MIN_CROP_PX or h_b < MIN_CROP_PX:
            return None

        crop = frame_bgr[y:y + h_b, x:x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    except Exception as exc:
        log.debug("align_face shim error: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Image storage helpers
# ══════════════════════════════════════════════════════════════════════════════

def _person_dir(employee_id: str) -> str:
    d = os.path.join(FACE_DB_DIR, employee_id)
    os.makedirs(os.path.join(d, "raw"),     exist_ok=True)
    os.makedirs(os.path.join(d, "aligned"), exist_ok=True)
    return d


def save_enrollment_crops(employee_id: str,
                           raw_crops:     list[np.ndarray],
                           aligned_crops: list[np.ndarray]) -> None:
    base = _person_dir(employee_id)
    ts   = int(time.time())
    for i, img in enumerate(raw_crops):
        if img is not None and img.size > 0:
            cv2.imwrite(os.path.join(base, "raw",     f"{ts}_{i}.jpg"), img)
    for i, img in enumerate(aligned_crops):
        if img is not None and img.size > 0:
            cv2.imwrite(os.path.join(base, "aligned", f"{ts}_{i}.jpg"), img)
    log.info("Saved %d raw + %d aligned crops for %s.",
             len(raw_crops), len(aligned_crops), employee_id)


def load_enrollment_crops(employee_id: str) -> list[np.ndarray]:
    aligned_dir = os.path.join(FACE_DB_DIR, employee_id, "aligned")
    if not os.path.isdir(aligned_dir):
        return []
    crops = []
    for fn in sorted(os.listdir(aligned_dir)):
        if fn.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(aligned_dir, fn))
            if img is not None:
                crops.append(img)
    return crops


# ══════════════════════════════════════════════════════════════════════════════
# Quality gate
# ══════════════════════════════════════════════════════════════════════════════

def quality_score(face_img: np.ndarray) -> float:
    """Laplacian variance → [0, 1] sharpness score."""
    if face_img is None or face_img.size == 0:
        return 0.0
    grey = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(grey, cv2.CV_64F).var()
    return float(np.clip((var - 20) / 180, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Identity cache with majority-vote buffer
# ══════════════════════════════════════════════════════════════════════════════

class IdentityCache:
    def __init__(self, ttl: float = IDENTITY_CACHE_TTL,
                 min_confidence: float = CONF_SKIP_THRESHOLD,
                 vote_window: int = VOTE_WINDOW):
        self._ttl      = ttl
        self._min_conf = min_confidence
        self._vw       = vote_window
        self._store: dict[str, dict] = {}
        self._lock  = threading.Lock()

    def get(self, track_uid: str) -> Optional[tuple[Optional[int], float]]:
        with self._lock:
            entry = self._store.get(track_uid)
        if entry is None:
            return None
        if (time.time() - entry["ts"]) >= self._ttl:
            return None
        if entry["conf"] >= self._min_conf:
            return entry["pid"], entry["conf"]
        return None

    def set(self, track_uid: str,
            person_id: Optional[int], confidence: float) -> None:
        with self._lock:
            if track_uid not in self._store:
                self._store[track_uid] = {
                    "pid": person_id, "conf": confidence,
                    "ts": time.time(), "votes": [],
                }
            entry = self._store[track_uid]
            votes: list = entry["votes"]
            votes.append((person_id, confidence))
            if len(votes) > self._vw:
                votes.pop(0)

            pid_counts = Counter(pid for pid, _ in votes)
            best_pid   = pid_counts.most_common(1)[0][0]
            best_conf  = float(np.mean(
                [conf for pid, conf in votes if pid == best_pid]))

            entry["pid"]  = best_pid
            entry["conf"] = best_conf
            entry["ts"]   = time.time()

    def invalidate(self, track_uid: str) -> None:
        with self._lock:
            self._store.pop(track_uid, None)

    def cleanup(self) -> None:
        cutoff = time.time() - self._ttl * 2
        with self._lock:
            self._store = {k: v for k, v in self._store.items()
                          if v["ts"] >= cutoff}


identity_cache = IdentityCache()


# ══════════════════════════════════════════════════════════════════════════════
# FaceRegistry
# ══════════════════════════════════════════════════════════════════════════════

class FaceRegistry:
    """
    Stateful face enrolment & recognition using InsightFace (ArcFace).

    Internals
    ─────────
    • _cache  : {person_id: [512-d np.float32, ...]}
    • recognize() uses vectorised cosine similarity (dot product on L2-normed vecs)
    • All embeddings stored as JSON in DB for portability
    """

    def __init__(self):
        os.makedirs(FACE_DB_DIR, exist_ok=True)
        self._cache : dict[int, list[np.ndarray]] = {}
        self._lock   = threading.Lock()
        self._load_all_embeddings()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_all_embeddings(self) -> None:
        """Load all ArcFace embeddings from DB into memory cache."""
        with self._lock:
            self._cache.clear()
        session = get_session()
        try:
            rows = session.query(FaceEmbedding).all()
            loaded, skipped = 0, 0
            for row in rows:
                vec = np.array(json.loads(row.embedding), dtype=np.float32)
                if vec.shape[0] not in (128, 512):
                    skipped += 1
                    continue
                # Skip old 128-d dlib embeddings — incompatible with ArcFace
                if vec.shape[0] == 128:
                    skipped += 1
                    log.warning(
                        "Skipping legacy 128-d dlib embedding (person_id=%d). "
                        "Re-enroll this person to use ArcFace.", row.person_id)
                    continue
                with self._lock:
                    self._cache.setdefault(row.person_id, []).append(vec)
                loaded += 1
            log.info(
                "Loaded %d ArcFace embeddings for %d persons (%d legacy skipped).",
                loaded, len(self._cache), skipped)
        except Exception as e:
            log.exception("Failed to load embeddings: %s", e)
        finally:
            session.close()

    def reload_cache(self) -> None:
        self._load_all_embeddings()

    def _is_duplicate(self, new_vec: np.ndarray,
                      existing_vecs: list[np.ndarray]) -> bool:
        """True if new_vec is too similar to any existing vector."""
        if not existing_vecs:
            return False
        mat  = np.array(existing_vecs, dtype=np.float32)
        sims = mat @ new_vec   # dot product = cosine sim (L2-normed)
        return bool(np.any(sims > DEDUP_SIMILARITY))

    # ── Augmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def augment_crops(crop: np.ndarray) -> list[np.ndarray]:
        """
        Return several augmented variants of a face crop.
        Augmentations help cover lighting/angle variation during enrollment.
        Note: for ArcFace, excessive augmentation hurts more than it helps —
        keep variants subtle (flip, mild brightness, light blur).
        """
        h, w = crop.shape[:2]
        if h < MIN_CROP_PX or w < MIN_CROP_PX:
            return [crop]

        results = [crop]

        # Horizontal flip (catches mirrored camera setups)
        results.append(cv2.flip(crop, 1))

        # CLAHE contrast normalisation
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            results.append(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                                        cv2.COLOR_LAB2BGR))
        except Exception:
            pass

        # Mild brightness adjustments
        for delta in (18, -18):
            results.append(
                np.clip(crop.astype(np.int16) + delta, 0, 255).astype(np.uint8))

        # Gentle Gaussian blur (simulate focus variation)
        results.append(cv2.GaussianBlur(crop, (3, 3), 0))

        return results

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll_person(
        self,
        name:        str,
        employee_id: str,
        role:        str,
        department:  str,
        face_crops:  list[np.ndarray],
        photo_path:  Optional[str] = None,
    ) -> Optional[Person]:
        """
        Enroll a new person from a list of raw BGR face crops or full frames.

        For each crop, extract_faces() is called to get the ArcFace embedding.
        If InsightFace fails to detect a face in a crop (e.g. the crop is
        already tightly cropped), a 112×112 resize is used as fallback and
        the embedding is extracted directly.
        """
        if not face_crops:
            log.error("No face crops provided for enrolment.")
            return None
        if not _INSIGHT_OK:
            log.error("InsightFace not available — cannot enroll.")
            return None

        session = get_session()
        try:
            if session.query(Person).filter_by(employee_id=employee_id).first():
                log.warning("Employee ID %s already exists.", employee_id)
                return None

            # ── Extract embeddings from each crop ──────────────────────────
            emb_crop_pairs: list[tuple[np.ndarray, np.ndarray]] = []

            for raw_crop in face_crops:
                if raw_crop is None or raw_crop.size == 0:
                    continue

                # Try full-frame detection first (best alignment)
                faces = extract_faces(raw_crop)
                if faces:
                    emb  = faces[0]["embedding"]
                    crop = raw_crop
                    emb_crop_pairs.append((emb, crop))
                    continue

                # Fallback: crop is already a tight face — resize and embed
                try:
                    resized = cv2.resize(raw_crop, (TARGET_SIZE, TARGET_SIZE),
                                        interpolation=cv2.INTER_LINEAR)
                    app = _get_insight_app()
                    if app is None:
                        continue
                    with _insight_lock:
                        faces_fb = app.get(resized)
                    if faces_fb:
                        emb_crop_pairs.append(
                            (faces_fb[0].normed_embedding.astype(np.float32),
                             resized))
                except Exception as exc:
                    log.debug("Fallback embedding failed: %s", exc)

            if not emb_crop_pairs:
                log.error("No embeddings extracted — enrollment aborted.")
                return None

            # ── Quality-rank and deduplicate ───────────────────────────────
            scored = sorted(
                emb_crop_pairs,
                key=lambda pair: quality_score(pair[1]),
                reverse=True,
            )[:ENROLL_SHOTS]

            person = Person(name=name, employee_id=employee_id,
                            role=role, department=department,
                            photo_path=photo_path)
            session.add(person)
            session.flush()

            pending       : list[np.ndarray] = []
            aligned_store : list[np.ndarray] = []

            for emb, crop in scored:
                # Augmentation loop for coverage
                for variant in self.augment_crops(crop):
                    if len(pending) >= MAX_EMBEDDINGS:
                        break
                    # Get embedding for this augmented variant
                    aug_faces = extract_faces(variant)
                    aug_emb   = aug_faces[0]["embedding"] if aug_faces else emb

                    if self._is_duplicate(aug_emb, pending):
                        continue
                    session.add(FaceEmbedding(
                        person_id  = person.id,
                        embedding  = json.dumps(aug_emb.tolist()),
                        model_name = MODEL_NAME,
                    ))
                    pending.append(aug_emb)
                    aligned_store.append(variant)

            if not pending:
                session.rollback()
                log.error("No unique embeddings extracted — enrolment aborted.")
                return None

            session.commit()
            session.refresh(person)

            raw_crops_to_save = [pair[1] for pair in scored]
            save_enrollment_crops(employee_id, raw_crops_to_save, aligned_store)

            with self._lock:
                self._cache.setdefault(person.id, []).extend(pending)

            log.info("Enrolled '%s' (%s) — %d ArcFace embeddings.",
                     name, employee_id, len(pending))
            return person

        except Exception as exc:
            session.rollback()
            log.exception("Enrolment failed: %s", exc)
            return None
        finally:
            session.close()

    # ── Add more embeddings to an existing person ──────────────────────────────

    def add_embeddings(self, person_id: int,
                       face_crops: list[np.ndarray]) -> int:
        with self._lock:
            existing = list(self._cache.get(person_id, []))

        if len(existing) >= MAX_EMBEDDINGS:
            log.info("Person %d already at max embeddings.", person_id)
            return 0

        session = get_session()
        added   = 0
        try:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person:
                return 0

            new_vecs      : list[np.ndarray] = []
            aligned_store : list[np.ndarray] = []
            pool = list(existing)

            for crop in face_crops:
                if crop is None or crop.size == 0:
                    continue
                faces = extract_faces(crop)
                if not faces:
                    continue

                for variant in self.augment_crops(crop):
                    if (len(existing) + len(new_vecs)) >= MAX_EMBEDDINGS:
                        break
                    aug_faces = extract_faces(variant)
                    aug_emb   = aug_faces[0]["embedding"] if aug_faces else faces[0]["embedding"]
                    if self._is_duplicate(aug_emb, pool + new_vecs):
                        continue
                    session.add(FaceEmbedding(
                        person_id  = person_id,
                        embedding  = json.dumps(aug_emb.tolist()),
                        model_name = MODEL_NAME,
                    ))
                    new_vecs.append(aug_emb)
                    aligned_store.append(variant)

            if new_vecs:
                session.commit()
                save_enrollment_crops(person.employee_id, face_crops, aligned_store)
                with self._lock:
                    self._cache.setdefault(person_id, []).extend(new_vecs)
                added = len(new_vecs)
                log.info("Added %d embedding(s) to person %d (total %d).",
                         added, person_id,
                         len(self._cache.get(person_id, [])))

        except Exception as exc:
            session.rollback()
            log.exception("add_embeddings failed: %s", exc)
        finally:
            session.close()
        return added

    # ── Recognition ───────────────────────────────────────────────────────────

    def recognize(self,
                  face_input,
                  ) -> tuple[Optional[Person], float]:
        """
        Identify a face.

        face_input can be:
          - np.ndarray of shape (512,)  — a pre-computed ArcFace embedding
          - np.ndarray of shape (H,W,3) — a BGR crop/frame (embedding extracted here)

        Returns (Person, confidence 0–1) or (None, 0.0).

        Pipeline
        ────────
        1.  Accept pre-computed embedding (fast path from camera worker) OR
            run extract_faces() on a raw crop (slow path from enrollment utils)
        2.  Vectorised cosine similarity against all enrolled embeddings
        3.  Best match wins if distance < TOLERANCE
        4.  Margin check: best person must be clearly better than second-best
        """
        # ── Resolve embedding ──────────────────────────────────────────────
        if face_input is None:
            return None, 0.0

        if isinstance(face_input, np.ndarray) and face_input.ndim == 1:
            # Pre-computed 512-d embedding (fast path)
            embedding = face_input.astype(np.float32)
        else:
            # Raw BGR image — extract embedding
            faces = extract_faces(face_input)
            if not faces:
                return None, 0.0
            embedding = faces[0]["embedding"]

        # ── Gallery snapshot ───────────────────────────────────────────────
        with self._lock:
            if not self._cache:
                return None, 0.0
            pids, vecs = [], []
            for pid, evecs in self._cache.items():
                for v in evecs:
                    pids.append(pid)
                    vecs.append(v)

        if not vecs:
            return None, 0.0

        mat  = np.array(vecs, dtype=np.float32)  # (N, 512)
        sims = mat @ embedding                    # cosine similarity (L2-normed)

        best_idx  = int(np.argmax(sims))
        best_sim  = float(sims[best_idx])
        best_pid  = pids[best_idx]
        best_dist = 1.0 - best_sim

        # ── Reject if above tolerance ──────────────────────────────────────
        if best_dist > TOLERANCE:
            return None, 0.0

        # ── Margin check: best vs second-best different person ─────────────
        other_mask = np.array([p != best_pid for p in pids])
        if other_mask.any():
            second_sim  = float(np.max(sims[other_mask]))
            second_dist = 1.0 - second_sim
            margin      = second_dist - best_dist
            if margin < 0.08:
                log.debug(
                    "Ambiguous recognition: best_dist=%.4f second_dist=%.4f gap=%.4f",
                    best_dist, second_dist, margin)
                return None, 0.0

        confidence = float(best_sim)  # cosine similarity ≈ 0.4–1.0

        session = get_session()
        try:
            person = session.query(Person).filter_by(id=best_pid).first()
            return person, confidence
        finally:
            session.close()

    # ── Deletion ───────────────────────────────────────────────────────────────

    def delete_person(self, person_id: int) -> bool:
        session = get_session()
        try:
            person = session.query(Person).filter_by(id=person_id).first()
            if not person:
                return False
            eid = person.employee_id
            session.delete(person)
            session.commit()
            with self._lock:
                self._cache.pop(person_id, None)
            import shutil
            person_dir = os.path.join(FACE_DB_DIR, eid)
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir, ignore_errors=True)
            return True
        except Exception as exc:
            session.rollback()
            log.exception("Delete failed: %s", exc)
            return False
        finally:
            session.close()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def embedding_stats(self) -> dict:
        session = get_session()
        try:
            result = {}
            with self._lock:
                cache_snap = dict(self._cache)
            for pid, vecs in cache_snap.items():
                p   = session.query(Person).filter_by(id=pid).first()
                eid = p.employee_id if p else str(pid)
                aligned_dir = os.path.join(FACE_DB_DIR, eid, "aligned")
                disk_count  = (len(os.listdir(aligned_dir))
                               if os.path.isdir(aligned_dir) else 0)
                result[pid] = {
                    "name":         p.name if p else "?",
                    "count":        len(vecs),
                    "embedding_dim": vecs[0].shape[0] if vecs else 0,
                    "disk_aligned": disk_count,
                    "backend":      "insightface_arcface",
                    "threshold":    TOLERANCE,
                }
            return result
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════════════════════
# One-time migration helper
# ══════════════════════════════════════════════════════════════════════════════

def clear_legacy_embeddings():
    """
    Delete all 128-d dlib embeddings from the database.
    Run this ONCE after switching to InsightFace, then re-enroll all persons.

    Usage (from Python shell or a migration script):
        from face_registry import clear_legacy_embeddings
        clear_legacy_embeddings()
    """
    session = get_session()
    try:
        rows    = session.query(FaceEmbedding).all()
        deleted = 0
        for row in rows:
            vec = np.array(json.loads(row.embedding), dtype=np.float32)
            if vec.shape[0] != 512:
                session.delete(row)
                deleted += 1
        session.commit()
        log.info("Deleted %d legacy embeddings.", deleted)
        print(f"Deleted {deleted} legacy 128-d dlib embeddings. Re-enroll all persons.")
    except Exception as exc:
        session.rollback()
        log.exception("Migration failed: %s", exc)
    finally:
        session.close()
