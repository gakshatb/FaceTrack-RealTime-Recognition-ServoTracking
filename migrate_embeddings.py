"""
migrate_embeddings.py
"""

import os
import logging
import cv2 # type: ignore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Imports ───────────────────────────────────────────────────────────────────
from database       import init_db, get_session, Person, FaceEmbedding
from face_registry  import FaceRegistry, FACE_DB_DIR, clear_legacy_embeddings

def main():
    init_db()
    session = get_session()

    print("\n" + "═" * 60)
    print("  FaceTrack — Embedding Migration (dlib → ArcFace)")
    print("═" * 60)

    # ── Step 1: Count existing embeddings ─────────────────────────────────────
    total_rows = session.query(FaceEmbedding).count()
    persons    = session.query(Person).all()
    session.close()

    print(f"\nFound {total_rows} embedding rows across {len(persons)} persons.")

    if total_rows == 0:
        print("Nothing to migrate. Database is empty.")
        return

    # ── Step 2: Clear all legacy embeddings ───────────────────────────────────
    print("\nStep 1/2 — Clearing legacy 128-d dlib embeddings…")
    clear_legacy_embeddings()

    # ── Step 3: Re-enroll from saved crops ────────────────────────────────────
    print("\nStep 2/2 — Re-enrolling persons from saved aligned crops…")
    registry   = FaceRegistry()
    ok_count   = 0
    fail_count = 0
    skip_count = 0

    for person in persons:
        eid         = person.employee_id
        aligned_dir = os.path.join(FACE_DB_DIR, eid, "aligned")

        if not os.path.isdir(aligned_dir):
            print(f"  ⚠  {person.name} ({eid}) — no aligned crops found, SKIP")
            print(f"     → Re-enroll manually via the web UI or upload new photos.")
            skip_count += 1
            continue

        crops = []
        for fn in sorted(os.listdir(aligned_dir)):
            if fn.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(aligned_dir, fn))
                if img is not None:
                    crops.append(img)

        if not crops:
            print(f"  ⚠  {person.name} ({eid}) — aligned dir empty, SKIP")
            skip_count += 1
            continue

        print(f"  →  {person.name} ({eid}) — {len(crops)} crops found…", end=" ")

        # Delete existing DB record and re-create (registry.enroll_person checks uniqueness)
        session2 = get_session()
        try:
            p_db = session2.query(Person).filter_by(employee_id=eid).first()
            if p_db:
                # Temporarily remove unique constraint block by clearing the person
                # then re-using their data
                name       = p_db.name
                role       = p_db.role
                department = p_db.department
                photo_path = p_db.photo_path
                is_active  = p_db.is_active
                # Delete embeddings only
                session2.query(FaceEmbedding).filter_by(person_id=p_db.id).delete()
                session2.commit()
        except Exception as ex:
            session2.rollback()
            log.warning("Failed to clear embeddings for %s: %s", eid, ex)
        finally:
            session2.close()

        # Now add new ArcFace embeddings
        session3 = get_session()
        try:
            p_db = session3.query(Person).filter_by(employee_id=eid).first()
            if not p_db:
                skip_count += 1
                print("DB record missing, SKIP")
                continue
            added = registry.add_embeddings(p_db.id, crops)
            if added > 0:
                print(f"OK ({added} ArcFace embeddings)")
                ok_count += 1
            else:
                print("FAILED — no embeddings extracted")
                fail_count += 1
        except Exception as ex:
            print(f"ERROR: {ex}")
            fail_count += 1
        finally:
            session3.close()

    print("\n" + "─" * 60)
    print(f"Migration complete.")
    print(f"  ✓ Re-enrolled: {ok_count}")
    print(f"  ✗ Failed:      {fail_count}")
    print(f"  ⚠ Skipped:    {skip_count}")
    if fail_count > 0 or skip_count > 0:
        print("\nFor failed/skipped persons, re-enroll manually:")
        print("  • Use /users in the web UI → Add Embeddings")
        print("  • Or POST new photos to /api/persons/enroll")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
