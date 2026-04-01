import streamlit as st
import cv2, copy, tempfile, os, numpy as np, pandas as pd
from pathlib import Path
import boto3
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hashlib import md5

# ── S3 Configuration ───────────────────────────────────────
BUCKET_NAME = "itg-telemetry-bucket"
REGION_NAME = "us-east-1"  # adjust if different
s3_client = boto3.client("s3", region_name=REGION_NAME)

def upload_to_s3(file_path_or_bytes, bucket, key):
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as f:
            s3_client.put_object(Bucket=bucket, Key=key, Body=f)
    else:  # bytes-like
        s3_client.put_object(Bucket=bucket, Key=key, Body=file_path_or_bytes)

def file_hash(path: Path):
    h = md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

# ── Streamlit Page Setup ─────────────────────────────────
st.set_page_config(page_title="Footfall Detection", layout="centered")
col1, col2 = st.columns([25, 15])
with col1:
    st.title("Footfall Detection")
with col2:
    st.image("logo.png", width=400)
st.caption("YOLOv8n · BoT-SORT · Fast ReID")

if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
if st.sidebar.button("Reset App"):
    import gc, glob
    for f in glob.glob("*_out.mp4"):
        try: os.remove(f)
        except: pass
    for f in glob.glob("*_h264.mp4"):
        try: os.remove(f)
        except: pass
    st.session_state.clear()
    gc.collect()
    st.rerun()

CONF_THRESHOLD       = st.sidebar.slider("YOLO Confidence",           0.1,  0.9,  0.40, 0.05)
ENTRY_LINE_Y_RATIO   = st.sidebar.slider("Entry Line Y (0=top)",      0.1,  0.6,  0.35, 0.05)
EXIT_LINE_Y_RATIO    = st.sidebar.slider("Exit Line Y  (0=top)",      0.4,  0.9,  0.65, 0.05)
SIMILARITY_THRESHOLD = st.sidebar.slider("ReID Similarity Threshold", 0.50, 0.99, 0.82, 0.01)
MIN_TRACK_FRAMES     = st.sidebar.slider("Min Track Frames",          3,    20,   8,    1)
FRAME_SKIP           = st.sidebar.slider("Process every N frames",    1,    5,    2,    1)

# ── ReID embedding functions ─────────────────────────────
def extract_embedding(crop):
    h, w = crop.shape[:2]
    if h < 20 or w < 10: return None
    crop = cv2.resize(crop, (64, 128))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    upper = hsv[:64, :]
    lower = hsv[64:, :]
    feats = []
    for part in [upper, lower]:
        hist = cv2.calcHist([part], [0, 1], None, [16, 16], [0, 180, 0, 256])
        feats.append(cv2.normalize(hist, hist).flatten())
    emb = np.concatenate(feats)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)

def pid_color(pid):
    np.random.seed(pid * 137)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))

def cosine_sim(a, b_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a.reshape(1, -1), b_matrix)[0]

# ── Pipeline function ────────────────────────────────────
def run_pipeline(video_path, cfg):
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")

    CONF_THR  = cfg["conf_thr"]
    SIM_THR   = cfg["sim_thr"]
    MIN_TF    = cfg["min_track_frames"]
    ENTRY_R   = cfg["entry_y"]
    EXIT_R    = cfg["exit_y"]
    SKIP      = cfg["frame_skip"]

    person_database  = {}
    next_pid         = [1]
    event_log        = []
    track_emb_buffer = defaultdict(list)
    track_identity   = {}
    track_similarity = {}
    track_zone         = {}
    track_zone_stable  = {}
    track_zone_counter = {}
    STABLE_FRAMES      = 3
    pid_last_event_frame = defaultdict(lambda: defaultdict(lambda: -9999))
    EVENT_COOLDOWN       = 45
    pid_last_confirmed_event = {}
    track_counted_entry = set()
    log_lines = []

    cap   = cv2.VideoCapture(str(video_path))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    entry_y = int(H * ENTRY_R)
    exit_y  = int(H * EXIT_R)
    out_path = Path(video_path).with_name(Path(video_path).stem + "_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # ── Matching / Identity ─────────────────────────────
    def match_or_register(emb, fn):
        if not person_database:
            pid = next_pid[0]; next_pid[0] += 1
            person_database[pid] = {"embeddings":[emb.copy()], "mean_embedding":emb.copy(),
                                    "first_seen_frame":fn, "last_seen_frame":fn,
                                    "visit_count":0, "appearance_count":1}
            return pid, 0.0
        pids  = list(person_database.keys())
        means = np.array([person_database[p]["mean_embedding"] for p in pids])
        sims  = cosine_sim(emb, means)
        bi    = int(np.argmax(sims)); bs = float(sims[bi]); bp = pids[bi]
        if bs >= SIM_THR:
            r = person_database[bp]
            r["last_seen_frame"] = fn; r["appearance_count"] += 1
            e = r["embeddings"]
            if len(e) < 20: e.append(emb.copy())
            else: e.pop(0); e.append(emb.copy())
            r["mean_embedding"] = np.mean(e, axis=0)
            return bp, bs
        pid = next_pid[0]; next_pid[0] += 1
        person_database[pid] = {"embeddings":[emb.copy()], "mean_embedding":emb.copy(),
                                "first_seen_frame":fn, "last_seen_frame":fn,
                                "visit_count":0, "appearance_count":1}
        return pid, bs

    def get_identity(tid, emb, fn):
        if tid in track_identity:
            return track_identity[tid]
        track_emb_buffer[tid].append(emb)
        if len(track_emb_buffer[tid]) < MIN_TF:
            return None
        me = np.mean(track_emb_buffer[tid], axis=0)
        me /= (np.linalg.norm(me) + 1e-8)
        pid, sim = match_or_register(me, fn)
        track_identity[tid] = pid
        track_similarity[tid] = sim
        return pid

    def zone_of(cy):
        if cy < entry_y: return "above"
        elif cy > exit_y:  return "below"
        else:              return "between"

    def fire_event(pid, event, fn):
        last_confirmed = pid_last_confirmed_event.get(pid)
        if event == "EXITED" and last_confirmed != "ENTERED": return False
        if fn - pid_last_event_frame[pid][event] < EVENT_COOLDOWN: return False
        pid_last_event_frame[pid][event] = fn
        pid_last_confirmed_event[pid]    = event
        if event == "ENTERED":
            track_counted_entry.add(pid)
            person_database[pid]["visit_count"] += 1
        event_log.append({"frame": fn, "person_id": pid, "event": event})
        return True

    def check_zone(tid, pid, cy, fn):
        if pid is None: return
        new_zone = zone_of(cy)
        if track_zone.get(tid) != new_zone:
            track_zone[tid] = new_zone
            track_zone_counter[tid] = 1
        else:
            track_zone_counter[tid] += 1
        if track_zone_counter[tid] < STABLE_FRAMES: return
        stable_zone = track_zone_stable.get(tid)
        confirmed_zone = new_zone
        event = None
        if stable_zone is None:
            if pid not in track_counted_entry: event = "ENTERED"
        else:
            if stable_zone == "above" and confirmed_zone in ("between", "below"): event = "ENTERED"
            elif stable_zone in ("between", "below") and confirmed_zone == "above": event = "EXITED"
            elif stable_zone == "between" and confirmed_zone == "below": event = "EXITED"
        track_zone_stable[tid] = confirmed_zone
        if event: fire_event(pid, event, fn)

    def annotate(frame, detections, fn):
        out = frame.copy()
        cv2.line(out, (0, entry_y), (W, entry_y), (0, 255, 0), 2)
        cv2.putText(out, "ENTRY", (8, entry_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.line(out, (0, exit_y), (W, exit_y), (0, 0, 255), 2)
        cv2.putText(out, "EXIT", (8, exit_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
        for (x1, y1, x2, y2, tid, pid, sim) in detections:
            color = pid_color(pid) if pid else (200,200,200)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            cx_, cy_ = (x1+x2)//2, (y1+y2)//2
            cv2.circle(out, (cx_,cy_), 4, color, -1)
            lbl = f"P#{pid} v:{person_database[pid].get('visit_count',0)} s:{sim:.2f}" if pid else f"T#{tid} buffering..."
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1,y1-th-6), (x1+tw+4,y1), color, -1)
            cv2.putText(out, lbl, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        hud = f"Frame {fn}   Unique Persons: {len(person_database)}"
        cv2.rectangle(out, (0,0), (len(hud)*9+10,26), (0,0,0), -1)
        cv2.putText(out, hud, (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0),1)
        return out

    # ── Processing loop ─────────────────────────────
    frame_ph = st.empty()
    prog     = st.progress(0.0)
    status   = st.empty()
    log_ph   = st.empty()
    UI_REFRESH_EVERY = 30
    fn = 0
    last_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if fn % SKIP == 0:
            results = yolo.track(frame, classes=[0], conf=CONF_THR, tracker="botsort.yaml", persist=True, verbose=False)
            last_detections = []
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    tid = int(boxes.id[i].cpu().item())
                    cy  = (y1 + y2) / 2
                    crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
                    emb  = extract_embedding(crop)
                    if emb is None: continue
                    pid = get_identity(tid, emb, fn)
                    check_zone(tid, pid, cy, fn)
                    last_detections.append((x1,y1,x2,y2,tid,pid,track_similarity.get(tid,0.0)))
        annotated = annotate(frame, last_detections, fn)
        writer.write(annotated)
        if fn % UI_REFRESH_EVERY == 0:
            prog.progress(min(fn/max(total,1),1.0))
            status.text(f"Frame {fn}/{total}  |  Persons: {len(person_database)} | Events: {len(event_log)}")
            frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            log_ph.text("\n".join([f"{'🟢' if e['event']=='ENTERED' else '🔴'} {e['event']} — Person #{e['person_id']}" for e in event_log[-10:]]) or "Waiting for zone crossings...")
        fn += 1

    cap.release()
    writer.release()
    prog.progress(1.0)
    status.success("Processing complete!")

    return {"output_path": out_path, "database": copy.deepcopy(person_database), "events": copy.deepcopy(event_log)}

# ── Analytics function ───────────────────────────────────
def plot_analytics(db, events):
    rows = [{"person_id": p, "first_seen_frame": r["first_seen_frame"],
             "last_seen_frame": r["last_seen_frame"],
             "appearance_count": r["appearance_count"],
             "visit_count": r["visit_count"]}
            for p, r in db.items()]
    df = pd.DataFrame(rows)
    if df.empty: return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Re-ID Analytics", fontsize=13, fontweight="bold")
    colors   = ["#%02x%02x%02x" % pid_color(p) for p in df["person_id"]]
    pids_str = df["person_id"].astype(str)
    axes[0, 0].bar(pids_str, df["visit_count"], color=colors)
    axes[0, 0].set_title("Visit Count per Person"); axes[0, 0].set_xlabel("Person ID")
    axes[0, 1].bar(pids_str, df["appearance_count"], color=colors)
    axes[0, 1].set_title("Appearance Frames per Person"); axes[0, 1].set_xlabel("Person ID")
    for _, row in df.iterrows():
        col = "#%02x%02x%02x" % pid_color(int(row["person_id"]))
        axes[1, 0].barh(f"P#{int(row['person_id'])}", row["last_seen_frame"]-row["first_seen_frame"],
                        left=row["first_seen_frame"], color=col, edgecolor="black", linewidth=0.5)
    axes[1, 0].set_title("Activity Timeline"); axes[1, 0].set_xlabel("Frame Number")
    if events:
        ev_df = pd.DataFrame(events)
        for ev, marker, color in [("ENTERED", "^", "green"), ("EXITED", "v", "red")]:
            sub = ev_df[ev_df["event"]==ev]
            if not sub.empty:
                axes[1, 1].scatter(sub["frame"], sub["person_id"], marker=marker, color=color, s=80, label=ev, zorder=3)
        axes[1, 1].legend()
    axes[1, 1].set_title("Entry / Exit Events"); axes[1, 1].set_xlabel("Frame Number")
    axes[1, 1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig

# ── Summary, Visit Table & Event Log ────────────────────
def show_summary_and_events(db, events):
    st.subheader("Summary")
    total_unique  = len(db)
    total_entries = sum(1 for e in events if e["event"] == "ENTERED")
    total_exits   = sum(1 for e in events if e["event"] == "EXITED")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Persons", total_unique)
    c2.metric("Total Entries",  total_entries)
    c3.metric("Total Exits",    total_exits)

    st.subheader("Visit Table")
    if db:
        rows = [{"Person ID": p,
                 "Visit Count": r["visit_count"],
                 "Appearance Frames": r["appearance_count"],
                 "First Seen Frame": r["first_seen_frame"],
                 "Last Seen Frame": r["last_seen_frame"]}
                for p, r in db.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No persons detected.")

    st.subheader("Event Log")
    if events:
        ev_df = pd.DataFrame(events)
        ev_df["event"] = ev_df["event"].apply(lambda x: f"{'🟢' if x == 'ENTERED' else '🔴'} {x}")
        ev_df.columns  = ["Frame", "Person ID", "Event"]
        st.dataframe(ev_df, use_container_width=True)
    else:
        st.info("No entry/exit events recorded.")

# ── Main UI ─────────────────────────────────────────────
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
    tmp_input.write(video_file.read())
    tmp_input.flush()
    tmp_input_path = Path(tmp_input.name)

    # ── Show input video ──────────────────────────────
    st.subheader("Input Video")
    st.video(str(tmp_input_path))

    # ── S3 key derivation ────────────────────────────
    vid_hash       = file_hash(tmp_input_path)
    s3_input_key   = f"inputs/{vid_hash}.mp4"
    s3_output_key  = f"outputs/{vid_hash}_out.mp4"

    cfg = {
        "conf_thr": CONF_THRESHOLD,
        "sim_thr": SIMILARITY_THRESHOLD,
        "min_track_frames": MIN_TRACK_FRAMES,
        "entry_y": ENTRY_LINE_Y_RATIO,
        "exit_y": EXIT_LINE_Y_RATIO,
        "frame_skip": FRAME_SKIP,
    }

    # ── Run Detection button ──────────────────────────
    if st.button("Run Detection"):
        # Check both input and output folders explicitly
        def key_exists(key):
            try:
                s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                return True
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return False
                raise  # re-raise unexpected errors

        input_exists  = key_exists(s3_input_key)
        output_exists = key_exists(s3_output_key)

        # Always run the full pipeline for complete UI output
        results = run_pipeline(tmp_input_path, cfg)
        st.session_state.result = results

        # Upload to S3 only if not already present — no duplicates
        if not input_exists:
            upload_to_s3(tmp_input_path, BUCKET_NAME, s3_input_key)
        if not output_exists:
            upload_to_s3(results["output_path"], BUCKET_NAME, s3_output_key)

        st.subheader("Output Video")
        st.video(str(results["output_path"]))

        show_summary_and_events(results["database"], results["events"])

        fig = plot_analytics(results["database"], results["events"])
        if fig:
            st.pyplot(fig)

    # ── If result already in session (page rerun) ─────
    elif st.session_state.result:
        results = st.session_state.result
        st.subheader("Output Video")
        st.video(str(results["output_path"]))
        show_summary_and_events(results["database"], results["events"])
        fig = plot_analytics(results["database"], results["events"])
        if fig:
            st.pyplot(fig)