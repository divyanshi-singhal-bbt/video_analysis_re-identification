import cv2
import torch
import numpy as np
import uuid
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from pinecone import Pinecone
from collections import defaultdict

# ── Models ─────────────────────────────────────────
yolo = YOLO('yolov8n.pt')

# FIX 1: Use osnet_x1_0 (full-size) instead of osnet_x0_25 (quarter-size)
# The x0.25 model simply doesn't have enough capacity to distinguish
# people of similar build/clothing in challenging stairwell lighting.
reid = torchreid.models.build_model(
    name='osnet_x1_0',        # ← was osnet_x0_25
    num_classes=1,
    pretrained=True
)
reid.eval()

# Slightly longer max_age so tracks survive brief occlusions
tracker = DeepSort(max_age=25, n_init=4, nn_budget=100)

# ── Pinecone ───────────────────────────────────────
pc = Pinecone(api_key="pcsk_6wQfHU_F948EotXhGn7UVyYbjjGW2hb6HuQeAsiHeoWAFTBiSN1byr6a3ZJUoR91b2naPX")
index = pc.Index("person-reid")

# ── Thresholds ─────────────────────────────────────
MATCH_THRESHOLD   = 0.72   # with x1_0 model, embeddings are tighter
NEW_PERSON_FLOOR  = 0.55   # below this = definitely new person

MIN_FRAMES_COMMIT = 12     # must see for 12 frames before assigning ID
EMB_BANK_SIZE     = 20     # diverse embedding bank per person
MAX_EMB_PER_TRACK = 10     # rolling buffer per track

# FIX 2: Spatial gating — don't match a re-entering person to someone
# who was last seen on the OPPOSITE side of the frame.
# A person exiting left cannot re-enter from the right immediately.
SPATIAL_GATE_PIXELS = 180  # max distance between exit & re-entry bbox center

# ── State ──────────────────────────────────────────
track_to_person   = {}     # track_id -> person_id  (IMMUTABLE once set)
track_embeddings  = {}     # track_id -> list of embeddings
track_frame_count = {}     # track_id -> int

person_emb_bank   = {}     # person_id -> [np.array, ...]
person_last_bbox  = {}     # person_id -> (cx, cy) when last seen  ← FIX 2
person_pinecone_counter = defaultdict(int)

all_person_ids = set()


# ── Embedding ──────────────────────────────────────
def get_embedding(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or (x2 - x1) < 20 or (y2 - y1) < 40:
        return None
    crop_r = cv2.resize(crop, (128, 256))
    tensor = torch.from_numpy(crop_r[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.no_grad():
        emb = reid(tensor.unsqueeze(0)).squeeze().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-6 else None


def cosine_sim(a, b):
    return float(np.dot(a, b))


def best_bank_score(query_emb, bank):
    if not bank:
        return -1.0
    return max(cosine_sim(query_emb, e) for e in bank)


def add_to_bank(person_id, new_emb):
    bank = person_emb_bank.setdefault(person_id, [])
    if bank:
        # Only add if diverse enough
        if max(cosine_sim(new_emb, e) for e in bank) > 0.94:
            return
    bank.append(new_emb.copy())
    if len(bank) > EMB_BANK_SIZE:
        bank.pop(0)


def push_to_pinecone(person_id):
    person_pinecone_counter[person_id] += 1
    if person_pinecone_counter[person_id] < 25:
        return
    person_pinecone_counter[person_id] = 0
    bank = person_emb_bank.get(person_id, [])
    if not bank:
        return
    avg = np.mean(bank, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 1e-6:
        avg /= norm
        index.upsert([(person_id, avg.tolist(), {"person_id": person_id})])


# ── Core ID assignment with spatial gating ─────────
def assign_person_id(avg_emb, query_cx, query_cy, active_person_ids):
    """
    Match avg_emb against all known person banks.
    FIX 2: Spatial gate — skip candidates whose last known position
    is too far from the current detection center.
    This prevents a man entering from the bottom being matched to
    a woman who exited from the top.
    """
    best_pid   = None
    best_score = -1.0

    for pid, bank in person_emb_bank.items():
        if pid in active_person_ids:
            continue

        # FIX 2: Spatial gate check
        if pid in person_last_bbox:
            lx, ly = person_last_bbox[pid]
            dist = ((query_cx - lx) ** 2 + (query_cy - ly) ** 2) ** 0.5
            if dist > SPATIAL_GATE_PIXELS:
                print(f"    Spatial gate blocked {pid} (dist={dist:.0f}px)")
                continue

        score = best_bank_score(avg_emb, bank)
        if score > best_score:
            best_score = score
            best_pid   = pid

    print(f"    Local best: {best_score:.3f} -> {best_pid}")

    if best_pid is not None and best_score >= MATCH_THRESHOLD:
        return best_pid, False

    # Pinecone fallback
    results = index.query(vector=avg_emb.tolist(), top_k=5, include_metadata=True)
    for match in results.matches:
        pid   = match.metadata['person_id']
        score = match.score
        if pid in active_person_ids:
            continue
        if pid in person_last_bbox:
            lx, ly = person_last_bbox[pid]
            dist = ((query_cx - lx) ** 2 + (query_cy - ly) ** 2) ** 0.5
            if dist > SPATIAL_GATE_PIXELS:
                continue
        if score > best_score:
            best_score = score
            best_pid   = pid

    print(f"    Pinecone best: {best_score:.3f} -> {best_pid}")

    if best_pid is not None and best_score >= MATCH_THRESHOLD:
        return best_pid, False

    # Soft match zone
    if best_pid is not None and best_score >= NEW_PERSON_FLOOR:
        print(f"    Soft accept {best_pid} ({best_score:.3f})")
        return best_pid, False

    # Genuinely new
    new_id = str(uuid.uuid4())[:8]
    index.upsert([(new_id, avg_emb.tolist(), {"person_id": new_id})])
    person_emb_bank[new_id] = [avg_emb.copy()]
    print(f"    🆕 New: {new_id}")
    return new_id, True


def resolve_conflicts(tracks):
    pid_to_tids = defaultdict(list)
    for t in tracks:
        if t.is_confirmed() and t.track_id in track_to_person:
            pid_to_tids[track_to_person[t.track_id]].append(t.track_id)
    for pid, tids in pid_to_tids.items():
        if len(tids) > 1:
            best = max(tids, key=lambda x: track_frame_count.get(x, 0))
            for tid in tids:
                if tid != best:
                    print(f"⚠️ Conflict {pid}: drop track {tid}")
                    del track_to_person[tid]
                    track_embeddings[tid] = []
                    track_frame_count[tid] = 0


# ── Video I/O ──────────────────────────────────────
VIDEO_PATH = "sample_video.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    dets    = yolo(frame)[0].boxes.data.cpu().numpy()
    persons = dets[(dets[:, 4] > 0.5) & (dets[:, 5] == 0)]

    raw_tracks = []
    for det in persons:
        emb = get_embedding(frame, det[:4])
        if emb is not None:
            raw_tracks.append((det[:4], det[4], emb))

    tracks = tracker.update_tracks(raw_tracks, frame=frame)

    if frame_idx % 5 == 0:
        resolve_conflicts(tracks)

    active_person_ids = {
        track_to_person[t.track_id]
        for t in tracks
        if t.is_confirmed() and t.track_id in track_to_person
    }

    for t in tracks:
        if not t.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, t.to_tlbr())
        tid = t.track_id
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        emb = get_embedding(frame, [x1, y1, x2, y2])
        if emb is None:
            continue

        buf = track_embeddings.setdefault(tid, [])
        buf.append(emb)
        if len(buf) > MAX_EMB_PER_TRACK:
            buf.pop(0)
        track_frame_count[tid] = track_frame_count.get(tid, 0) + 1

        avg_emb = np.mean(buf, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm < 1e-6:
            continue
        avg_emb /= norm

        if tid not in track_to_person:
            if track_frame_count[tid] < MIN_FRAMES_COMMIT:
                # Show grey box with "?" while waiting
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(frame, "?", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                continue

            person_id, is_new = assign_person_id(avg_emb, cx, cy, active_person_ids)

            # FIX 3: FREEZE the assignment — once a track gets an ID it NEVER changes
            track_to_person[tid] = person_id
            active_person_ids.add(person_id)
            all_person_ids.add(person_id)

            # Seed the bank with all buffered embeddings
            for e in buf:
                add_to_bank(person_id, e)
            # Record spatial position
            person_last_bbox[person_id] = (cx, cy)

            print(f"{'🆕 New' if is_new else '🔄 ReID'}: track {tid} -> {person_id}")
            label = person_id

        else:
            # FIX 3: ID is frozen, never re-queried — just use what was assigned
            person_id = track_to_person[tid]
            label = person_id

            # Grow bank and update last-seen position
            add_to_bank(person_id, emb)
            person_last_bbox[person_id] = (cx, cy)   # ← keep spatial position fresh
            push_to_pinecone(person_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)

    count_text = f"People: {len(all_person_ids)}"
    cv2.rectangle(frame, (8, 8), (180, 38), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Done. Total unique people: {len(all_person_ids)}")
print(f"   IDs: {sorted(all_person_ids)}")