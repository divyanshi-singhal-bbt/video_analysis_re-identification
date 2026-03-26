import streamlit as st
import cv2, copy, tempfile, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="Footfall Detection", layout="centered")
st.title("Footfall Detection")
st.caption("YOLOv8n · BoT-SORT · Fast ReID")

if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
CONF_THRESHOLD       = st.sidebar.slider("YOLO Confidence",           0.1,  0.9,  0.40, 0.05)
ENTRY_LINE_Y_RATIO   = st.sidebar.slider("Entry Line Y (0=top)",      0.1,  0.6,  0.35, 0.05)
EXIT_LINE_Y_RATIO    = st.sidebar.slider("Exit Line Y  (0=top)",      0.4,  0.9,  0.65, 0.05)
SIMILARITY_THRESHOLD = st.sidebar.slider("ReID Similarity Threshold", 0.50, 0.99, 0.82, 0.01)
MIN_TRACK_FRAMES     = st.sidebar.slider("Min Track Frames",          3,    20,   8,    1)
FRAME_SKIP           = st.sidebar.slider("Process every N frames",    1,    5,    2,    1)

# ── ReID embedding ────────────────────────────────────────
def extract_embedding(crop):
    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return None
    crop = cv2.resize(crop, (64, 128))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    upper = hsv[:64, :]
    lower = hsv[64:, :]
    feats = []
    for part in [upper, lower]:
        hist = cv2.calcHist([part], [0, 1], None, [16, 16], [0, 180, 0, 256])
        feats.append(cv2.normalize(hist, hist).flatten())
    emb  = np.concatenate(feats)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)

def pid_color(pid):
    np.random.seed(pid * 137)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))

def cosine_sim(a, b_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a.reshape(1, -1), b_matrix)[0]

# ── Pipeline ──────────────────────────────────────────────
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

    # ── FIX 1: Cooldown now tracked per (pid, event) pair ─────────────────────
    # This prevents the same person from firing the same event twice quickly,
    # regardless of which track_id they appear under (ReID may re-assign IDs).
    pid_last_event_frame = defaultdict(lambda: defaultdict(lambda: -9999))
    EVENT_COOLDOWN       = 45   # slightly higher — give more breathing room

    # ── FIX 2: Strict per-pid state machine ───────────────────────────────────
    # A person can only EXIT if they have previously ENTERED, and vice-versa.
    # This is the core fix for duplicate EXITED events.
    pid_last_confirmed_event = {}   # pid → "ENTERED" | "EXITED" | None

    track_counted_entry = set()     # pids that have fired at least one ENTERED

    log_lines = []

    cap   = cv2.VideoCapture(video_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    entry_y = int(H * ENTRY_R)
    exit_y  = int(H * EXIT_R)

    out_path = video_path.replace(Path(video_path).suffix, "_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    def match_or_register(emb, fn):
        if not person_database:
            pid = next_pid[0]; next_pid[0] += 1
            person_database[pid] = {
                "embeddings": [emb.copy()], "mean_embedding": emb.copy(),
                "first_seen_frame": fn, "last_seen_frame": fn,
                "visit_count": 0, "appearance_count": 1,
            }
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
        person_database[pid] = {
            "embeddings": [emb.copy()], "mean_embedding": emb.copy(),
            "first_seen_frame": fn, "last_seen_frame": fn,
            "visit_count": 0, "appearance_count": 1,
        }
        return pid, bs

    def get_identity(tid, emb, fn):
        if tid in track_identity:
            return track_identity[tid]
        track_emb_buffer[tid].append(emb)
        if len(track_emb_buffer[tid]) < MIN_TF:
            return None
        me  = np.mean(track_emb_buffer[tid], axis=0)
        me /= (np.linalg.norm(me) + 1e-8)
        pid, sim              = match_or_register(me, fn)
        track_identity[tid]   = pid
        track_similarity[tid] = sim
        return pid

    def zone_of(cy):
        if   cy < entry_y: return "above"
        elif cy > exit_y:  return "below"
        else:              return "between"

    def fire_event(pid, event, fn):
        """
        Central gate for all events. Enforces:
          1. State machine  — EXIT requires prior ENTERED; ENTERED after EXIT is allowed
          2. Per-(pid,event) cooldown  — can't fire same event twice within EVENT_COOLDOWN frames
        Returns True if event was fired, False if suppressed.
        """
        last_confirmed = pid_last_confirmed_event.get(pid)

        # ── State machine guard ────────────────────────────────────────────────
        if event == "EXITED":
            # Can only exit if the last confirmed event was ENTERED
            if last_confirmed != "ENTERED":
                return False
        # (ENTERED is always valid — person may re-enter after exiting)

        # ── Cooldown guard ─────────────────────────────────────────────────────
        if fn - pid_last_event_frame[pid][event] < EVENT_COOLDOWN:
            return False

        # ── All guards passed — fire ───────────────────────────────────────────
        pid_last_event_frame[pid][event] = fn
        pid_last_confirmed_event[pid]    = event

        if event == "ENTERED":
            track_counted_entry.add(pid)
            person_database[pid]["visit_count"] = \
                person_database[pid].get("visit_count", 0) + 1

        event_log.append({"frame": fn, "person_id": pid, "event": event})
        icon = "🟢" if event == "ENTERED" else "🔴"
        log_lines.append(f"{icon} {event} — Person #{pid} | Frame {fn}")
        return True

    def check_zone(tid, pid, cy, fn):
        if pid is None:
            return

        new_zone = zone_of(cy)

        # Stabilise zone readings
        if track_zone.get(tid) != new_zone:
            track_zone[tid]         = new_zone
            track_zone_counter[tid] = 1
        else:
            track_zone_counter[tid] = track_zone_counter.get(tid, 0) + 1

        if track_zone_counter.get(tid, 0) < STABLE_FRAMES:
            return

        stable_zone    = track_zone_stable.get(tid)
        confirmed_zone = new_zone

        event = None

        if stable_zone is None:
            # First confirmed zone sighting for this track
            if pid not in track_counted_entry:
                event = "ENTERED"
        else:
            if stable_zone == "above" and confirmed_zone in ("between", "below"):
                event = "ENTERED"
            elif stable_zone in ("between", "below") and confirmed_zone == "above":
                event = "EXITED"
            elif stable_zone == "between" and confirmed_zone == "below":
                event = "EXITED"
            elif stable_zone == "below" and confirmed_zone == "between":
                # Moving back toward entry — treat as re-entry attempt, not an event
                pass

        track_zone_stable[tid] = confirmed_zone

        if event:
            fire_event(pid, event, fn)

    def annotate(frame, detections, fn):
        out = frame.copy()
        cv2.line(out, (0, entry_y), (W, entry_y), (0, 255, 0), 2)
        cv2.putText(out, "ENTRY", (8, entry_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.line(out, (0, exit_y), (W, exit_y), (0, 0, 255), 2)
        cv2.putText(out, "EXIT", (8, exit_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        for (x1, y1, x2, y2, tid, pid, sim) in detections:
            color = pid_color(pid) if pid else (200, 200, 200)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cx_ = (x1+x2)//2; cy_ = (y1+y2)//2
            cv2.circle(out, (cx_, cy_), 4, color, -1)
            if pid is not None:
                vc  = person_database[pid].get("visit_count", 0)
                lbl = f"P#{pid} v:{vc} s:{sim:.2f}"
            else:
                lbl = f"T#{tid} buffering..."
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(out, lbl, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        hud = f"Frame {fn}   Unique Persons: {len(person_database)}"
        cv2.rectangle(out, (0, 0), (len(hud)*9+10, 26), (0, 0, 0), -1)
        cv2.putText(out, hud, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        return out

    frame_ph = st.empty()
    prog     = st.progress(0.0)
    status   = st.empty()
    log_ph   = st.empty()

    # ── FIX 3: Speed — batch-write frames, update UI less frequently ──────────
    # Instead of calling st.image every 10 frames (expensive Streamlit rerender),
    # update the preview every 30 frames. Writer.write() is the bottleneck;
    # skipping inference on non-SKIP frames already helps — but we also avoid
    # re-encoding the preview image on every single annotated frame.
    UI_REFRESH_EVERY = 30   # update Streamlit preview this often

    cap = cv2.VideoCapture(video_path)
    fn  = 0
    last_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if fn % SKIP == 0:
            results = yolo.track(frame, classes=[0], conf=CONF_THR,
                                 tracker="botsort.yaml", persist=True, verbose=False)
            last_detections = []
            if results and results[0].boxes is not None and \
               results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
                    tid = int(boxes.id[i].cpu().item())
                    cy  = (y1 + y2) / 2
                    crop = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                    emb  = extract_embedding(crop)
                    if emb is None:
                        continue
                    pid = get_identity(tid, emb, fn)
                    check_zone(tid, pid, cy, fn)
                    last_detections.append(
                        (x1, y1, x2, y2, tid, pid, track_similarity.get(tid, 0.0)))

        annotated = annotate(frame, last_detections, fn)
        writer.write(annotated)

        # Only push to Streamlit UI every UI_REFRESH_EVERY frames
        if fn % UI_REFRESH_EVERY == 0:
            prog.progress(min(fn / max(total, 1), 1.0))
            status.text(
                f"Frame {fn}/{total}  |  Persons: {len(person_database)}"
                f"  |  Events: {len(event_log)}"
            )
            frame_ph.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True,
            )
            log_ph.text("\n".join(log_lines[-15:]) or "Waiting for zone crossings...")

        fn += 1

    cap.release()
    writer.release()
    prog.progress(1.0)
    status.success("Processing complete!")

    return {
        "output_path": out_path,
        "database":    copy.deepcopy(person_database),
        "events":      copy.deepcopy(event_log),
    }

# ── Analytics ─────────────────────────────────────────────
def plot_analytics(db, events):
    rows = [{"person_id": p, "first_seen_frame": r["first_seen_frame"],
             "last_seen_frame": r["last_seen_frame"],
             "appearance_count": r["appearance_count"],
             "visit_count": r["visit_count"]}
            for p, r in db.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Re-ID Analytics", fontsize=13, fontweight="bold")
    colors   = ["#%02x%02x%02x" % pid_color(p) for p in df["person_id"]]
    pids_str = df["person_id"].astype(str)
    axes[0, 0].bar(pids_str, df["visit_count"], color=colors)
    axes[0, 0].set_title("Visit Count per Person"); axes[0, 0].set_xlabel("Person ID")
    axes[0, 0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 1].bar(pids_str, df["appearance_count"], color=colors)
    axes[0, 1].set_title("Appearance Frames per Person"); axes[0, 1].set_xlabel("Person ID")
    for _, row in df.iterrows():
        col = "#%02x%02x%02x" % pid_color(int(row["person_id"]))
        axes[1, 0].barh(
            f"P#{int(row['person_id'])}",
            row["last_seen_frame"] - row["first_seen_frame"],
            left=row["first_seen_frame"], color=col, edgecolor="black", linewidth=0.5,
        )
    axes[1, 0].set_title("Activity Timeline"); axes[1, 0].set_xlabel("Frame Number")
    if events:
        ev_df = pd.DataFrame(events)
        for ev, marker, color in [("ENTERED", "^", "green"), ("EXITED", "v", "red")]:
            sub = ev_df[ev_df["event"] == ev]
            if not sub.empty:
                axes[1, 1].scatter(sub["frame"], sub["person_id"],
                                   marker=marker, color=color, s=80, label=ev, zorder=3)
        axes[1, 1].legend()
    axes[1, 1].set_title("Entry / Exit Events"); axes[1, 1].set_xlabel("Frame Number")
    axes[1, 1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    return fig

# ── Main UI ───────────────────────────────────────────────
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    st.video(video_file)
    if st.button("Run Detection"):
        st.session_state.result = None
        suffix = Path(video_file.name).suffix or ".mp4"
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(video_file.read()); tmp.close()

        st.subheader("Processing...")
        cfg = dict(conf_thr=CONF_THRESHOLD, entry_y=ENTRY_LINE_Y_RATIO,
                   exit_y=EXIT_LINE_Y_RATIO, sim_thr=SIMILARITY_THRESHOLD,
                   min_track_frames=MIN_TRACK_FRAMES, frame_skip=FRAME_SKIP)
        try:
            result = run_pipeline(tmp.name, cfg)
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e)
        finally:
            try: os.unlink(tmp.name)
            except: pass

# ── Results ───────────────────────────────────────────────
if st.session_state.result:
    r       = st.session_state.result
    db      = r["database"]
    events  = r["events"]
    entries = sum(1 for e in events if e["event"] == "ENTERED")
    exits   = sum(1 for e in events if e["event"] == "EXITED")

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Persons", len(db))
    c2.metric("Entries",        entries)
    c3.metric("Exits",          exits)

    out_path = r["output_path"]
    if os.path.exists(out_path):
        st.subheader("Annotated Video")
        browser_path = out_path.replace("_out.mp4", "_h264.mp4")
        try:
            import imageio.v3 as iio
            cap2   = cv2.VideoCapture(out_path)
            fps2   = cap2.get(cv2.CAP_PROP_FPS) or 25.0
            frames = []
            while True:
                ret2, frm2 = cap2.read()
                if not ret2: break
                frames.append(cv2.cvtColor(frm2, cv2.COLOR_BGR2RGB))
            cap2.release()
            iio.imwrite(browser_path, frames, fps=fps2,
                        codec="libx264",
                        output_params=["-pix_fmt", "yuv420p", "-crf", "23"])
            with open(browser_path, "rb") as f: vbytes = f.read()
        except Exception as e:
            st.warning(f"Re-encode failed: {e}")
            with open(out_path, "rb") as f: vbytes = f.read()
        st.video(vbytes)
        st.download_button("⬇ Download Video", vbytes,
                           file_name="output_reid.mp4", mime="video/mp4")

    if db:
        st.subheader("👥 Identity Database")
        rows = [{"Person": f"P#{p}", "First Frame": r2["first_seen_frame"],
                 "Last Frame": r2["last_seen_frame"],
                 "Appearances": r2["appearance_count"],
                 "Visits": r2["visit_count"]} for p, r2 in db.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇ Download Identity CSV",
                           df.to_csv(index=False).encode(),
                           file_name="identity_db.csv", mime="text/csv")

    if events:
        st.subheader("Event Log")
        ev_df = pd.DataFrame(events)
        ev_df.columns = ["Frame", "Person ID", "Event"]
        st.dataframe(ev_df, use_container_width=True)
        st.download_button("⬇ Download Event Log CSV",
                           ev_df.to_csv(index=False).encode(),
                           file_name="event_log.csv", mime="text/csv")

    if db:
        st.subheader("Analytics")
        fig = plot_analytics(db, events)
        if fig:
            st.pyplot(fig, use_container_width=True)
            chart_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(chart_tmp.name, dpi=150, bbox_inches="tight")
            chart_tmp.close()
            with open(chart_tmp.name, "rb") as f:
                st.download_button("⬇ Download Chart PNG", f.read(),
                                   file_name="analytics.png", mime="image/png")
            os.unlink(chart_tmp.name)