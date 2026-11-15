
import os
import csv
import cv2
import time
import uuid
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import mediapipe as mp

from flask import (
    Flask, Response, jsonify, render_template,
    send_file, abort, request
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0

EAR_THRESHOLD = 0.23
DROWSY_SECONDS = 2.0

MAR_THRESHOLD = 0.55
YAWN_SECONDS = 1.2

PERCLOS_WINDOW = 60.0
BLINK_WINDOW = 60.0
MIN_BLINK_DURATION = 0.08

ALARM_ENABLED = True

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [13, 14, 82, 87, 312, 317, 78, 308]


# ============================================================
# 2. FLASK AND SHARED STATE
# ============================================================

app = Flask(__name__)

lock = threading.RLock()
stop_event = threading.Event()

worker_thread = None
latest_frame = None

state = {
    "running": False,
    "status": "Idle",
    "message": "Ready to start monitoring",

    "ear": 0.0,
    "mar": 0.0,
    "perclos": 0.0,

    "session_id": None,
    "session_started": None,
    "session_elapsed": 0.0,

    "drowsiness_count": 0,
    "drowsy_duration": 0.0,

    "blink_count": 0,
    "avg_blink_duration": 0.0,
    "blink_rate": 0.0,
    "current_eye_closure": 0.0,
    "longest_eye_closure": 0.0,

    "yawn_count": 0,
    "max_perclos": 0.0,
    "valid_monitoring_time": 0.0,

    "face_detected": False,
    "last_alert": None,

    "events": [],
    "series": [],

    "error": None
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def eye_aspect_ratio(points, indices, width, height):

    pts = []

    for idx in indices:
        p = points[idx]

        pts.append(
            np.array(
                [p.x * width, p.y * height],
                dtype=np.float32
            )
        )

    p1, p2, p3, p4, p5, p6 = pts

    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return 0.0

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return float(ear)


def mouth_aspect_ratio(points, width, height):

    pts = {}

    for idx in MOUTH:
        p = points[idx]

        pts[idx] = np.array(
            [p.x * width, p.y * height],
            dtype=np.float32
        )

    vertical = (
        np.linalg.norm(pts[13] - pts[14]) +
        np.linalg.norm(pts[82] - pts[87]) +
        np.linalg.norm(pts[312] - pts[317])
    ) / 3.0

    horizontal = np.linalg.norm(
        pts[78] - pts[308]
    )

    if horizontal < 1e-6:
        return 0.0

    return float(vertical / horizontal)


def append_csv(path, headers, row):
    """
    Appends CSV data and supports older CSV schemas.
    """

    path = Path(path)

    if path.exists() and path.stat().st_size > 0:

        with path.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            old_headers = reader.fieldnames or []
            old_rows = list(reader)

        if old_headers != list(headers):

            merged_headers = list(
                dict.fromkeys(
                    list(old_headers) + list(headers)
                )
            )

            with path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=merged_headers
                )

                writer.writeheader()
                writer.writerows(old_rows)

            headers = merged_headers

    with path.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers,
            extrasaction="ignore"
        )

        if path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def add_event(event_type, message, value=None):

    event = {
        "time": now_iso(),
        "type": event_type,
        "message": message,
        "value": round(float(value), 3)
        if value is not None else None
    }

    with lock:

        state["events"].insert(0, event)
        state["events"] = state["events"][:100]

        if event_type in ("Drowsiness", "Yawn"):
            state["last_alert"] = event["time"]

        sid = state.get("session_id")

    if sid:

        append_csv(
            LOG_DIR / f"events_{sid}.csv",
            ["time", "type", "message", "value"],
            event
        )


def play_alarm():

    if not ALARM_ENABLED:
        return

    try:
        import winsound
        winsound.Beep(1500, 450)

    except Exception:
        pass


# ============================================================
# 4. SESSION MANAGEMENT
# ============================================================

def reset_session():

    sid = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:5]
    )

    with lock:

        state.update({

            "status": "Starting",
            "message": "Initializing camera and face landmark model",

            "ear": 0.0,
            "mar": 0.0,
            "perclos": 0.0,

            "session_id": sid,
            "session_started": time.time(),
            "session_elapsed": 0.0,

            "drowsiness_count": 0,
            "drowsy_duration": 0.0,

            "blink_count": 0,
            "avg_blink_duration": 0.0,
            "blink_rate": 0.0,
            "current_eye_closure": 0.0,
            "longest_eye_closure": 0.0,

            "yawn_count": 0,
            "max_perclos": 0.0,
            "valid_monitoring_time": 0.0,

            "face_detected": False,
            "last_alert": None,

            "events": [],
            "series": [],

            "error": None
        })

    return sid


def save_session_summary():

    with lock:

        sid = state.get("session_id")
        started = state.get("session_started")

        if not sid or not started:
            return

        elapsed = max(
            0.0,
            time.time() - started
        )

        row = {

            "session_id": sid,

            "date": datetime.fromtimestamp(
                started
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "duration_seconds": round(elapsed, 2),

            "valid_monitoring_seconds": round(
                state["valid_monitoring_time"], 2
            ),

            "drowsiness_events": state["drowsiness_count"],

            "drowsy_duration_seconds": round(
                state["drowsy_duration"], 2
            ),

            "blink_count": state["blink_count"],

            "average_blink_duration_seconds": round(
                state["avg_blink_duration"], 3
            ),

            "longest_eye_closure_seconds": round(
                state["longest_eye_closure"], 3
            ),

            "yawns": state["yawn_count"],

            "max_perclos_percent": round(
                state["max_perclos"], 2
            )
        }

    append_csv(
        LOG_DIR / "session_summary.csv",

        [
            "session_id",
            "date",
            "duration_seconds",
            "valid_monitoring_seconds",
            "drowsiness_events",
            "drowsy_duration_seconds",
            "blink_count",
            "average_blink_duration_seconds",
            "longest_eye_closure_seconds",
            "yawns",
            "max_perclos_percent"
        ],

        row
    )


# ============================================================
# 5. MAIN DETECTION WORKER
# ============================================================

def detection_worker():

    global latest_frame

    reset_session()

    cap = None
    landmarker = None

    session_start = time.time()

    closure_start = None

    blink_durations = []
    blink_timestamps = deque()

    yawn_start = None

    drowsy_active = False
    yawn_active = False

    perclos_samples = deque()

    last_series_write = 0.0
    last_alarm = 0.0

    drowsy_accumulated = 0.0

    try:

        # Model validation
        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Face Landmarker model not found: {MODEL_PATH}. "
                "Place face_landmarker.task inside the models folder."
            )

        # Camera initialization
        cap = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        )

        if not cap.isOpened():

            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():

            raise RuntimeError(
                "Could not open webcam. Close other camera apps "
                "or change CAMERA_INDEX."
            )

        # MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions

        FaceLandmarker = (
            mp.tasks.vision.FaceLandmarker
        )

        FaceLandmarkerOptions = (
            mp.tasks.vision.FaceLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = FaceLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),

            running_mode=RunningMode.VIDEO,

            num_faces=1,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False
        )

        landmarker = (
            FaceLandmarker.create_from_options(options)
        )

        with lock:

            state["status"] = "Monitoring"
            state["message"] = "Face monitoring is active"

        add_event(
            "System",
            "Monitoring session started"
        )

        # ----------------------------------------------------
        # FRAME PROCESSING LOOP
        # ----------------------------------------------------

        while not stop_event.is_set():

            ok, frame = cap.read()

            if not ok:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - session_start) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            current_time = time.monotonic()

            elapsed = time.time() - session_start

            ear = 0.0
            mar = 0.0
            perclos = 0.0

            face_found = bool(
                result.face_landmarks
            )

            # =================================================
            # FACE DETECTED
            # =================================================

            if face_found:

                points = result.face_landmarks[0]

                left_ear = eye_aspect_ratio(
                    points,
                    LEFT_EYE,
                    width,
                    height
                )

                right_ear = eye_aspect_ratio(
                    points,
                    RIGHT_EYE,
                    width,
                    height
                )

                ear = (
                    left_ear + right_ear
                ) / 2.0

                mar = mouth_aspect_ratio(
                    points,
                    width,
                    height
                )

                # ---------------------------------------------
                # PERCLOS
                # ---------------------------------------------

                closed = ear < EAR_THRESHOLD

                perclos_samples.append(
                    (current_time, 1 if closed else 0)
                )

                while (
                    perclos_samples
                    and current_time - perclos_samples[0][0]
                    > PERCLOS_WINDOW
                ):

                    perclos_samples.popleft()

                perclos = (

                    sum(x[1] for x in perclos_samples)
                    / len(perclos_samples) * 100.0

                    if perclos_samples else 0.0
                )

                # ---------------------------------------------
                # EYE CLOSURE AND BLINK DETECTION
                # ---------------------------------------------

                if closed:

                    if closure_start is None:
                        closure_start = current_time

                    closed_duration = (
                        current_time - closure_start
                    )

                    with lock:

                        state["current_eye_closure"] = (
                            closed_duration
                        )

                    # Prolonged closure / drowsiness
                    if (
                        closed_duration >= DROWSY_SECONDS
                        and not drowsy_active
                    ):

                        drowsy_active = True

                        with lock:
                            state["drowsiness_count"] += 1

                        add_event(
                            "Drowsiness",
                            "Prolonged eye closure detected",
                            closed_duration
                        )

                        play_alarm()

                        last_alarm = current_time

                    # Repeat alarm while prolonged closure continues
                    if (
                        drowsy_active
                        and current_time - last_alarm >= 3.0
                    ):

                        play_alarm()
                        last_alarm = current_time

                else:

                    if closure_start is not None:

                        closure_duration = max(
                            0.0,
                            current_time - closure_start
                        )

                        # Completed eye closure cycle
                        if closure_duration >= MIN_BLINK_DURATION:

                            blink_durations.append(
                                closure_duration
                            )

                            blink_timestamps.append(
                                current_time
                            )

                            with lock:

                                state["blink_count"] += 1

                                state["avg_blink_duration"] = (
                                    sum(blink_durations)
                                    / len(blink_durations)
                                )

                                state["longest_eye_closure"] = max(
                                    state["longest_eye_closure"],
                                    closure_duration
                                )

                            add_event(
                                "Blink",
                                "Eye closure cycle completed",
                                closure_duration
                            )

                        if drowsy_active:

                            drowsy_accumulated += (
                                closure_duration
                            )

                    closure_start = None
                    drowsy_active = False

                    with lock:
                        state["current_eye_closure"] = 0.0

                # ---------------------------------------------
                # BLINK FREQUENCY: ROLLING 60 SECONDS
                # ---------------------------------------------

                while (
                    blink_timestamps
                    and current_time - blink_timestamps[0]
                    > BLINK_WINDOW
                ):

                    blink_timestamps.popleft()

                with lock:

                    state["blink_rate"] = float(
                        len(blink_timestamps)
                    )

                # ---------------------------------------------
                # YAWN DETECTION
                # ---------------------------------------------

                mouth_open = mar > MAR_THRESHOLD

                if mouth_open:

                    if yawn_start is None:
                        yawn_start = current_time

                    if (
                        current_time - yawn_start >= YAWN_SECONDS
                        and not yawn_active
                    ):

                        yawn_active = True

                        with lock:
                            state["yawn_count"] += 1

                        add_event(
                            "Yawn",
                            "Possible yawn detected",
                            mar
                        )

                else:

                    yawn_start = None
                    yawn_active = False

                # Valid monitoring time
                fps = cap.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 30.0

                with lock:

                    state["valid_monitoring_time"] += (
                        1.0 / fps
                    )

            # =================================================
            # FACE NOT DETECTED
            # =================================================

            else:

                # Missing face is not counted as eye closure.
                perclos_samples.clear()

                closure_start = None

                drowsy_active = False

                yawn_start = None
                yawn_active = False

                with lock:

                    state["current_eye_closure"] = 0.0

                    state["blink_rate"] = float(
                        len(blink_timestamps)
                    )

            # =================================================
            # UPDATE SHARED STATE
            # =================================================

            with lock:

                state["ear"] = ear
                state["mar"] = mar
                state["perclos"] = perclos

                state["max_perclos"] = max(
                    state["max_perclos"],
                    perclos
                )

                state["face_detected"] = face_found

                state["session_elapsed"] = elapsed

                if face_found:

                    state["status"] = (
                        "Drowsiness Alert"
                        if drowsy_active
                        else "Monitoring"
                    )

                    state["message"] = (
                        "Prolonged eye closure detected"
                        if drowsy_active
                        else "Face detected — monitoring active"
                    )

                else:

                    state["status"] = "Face Not Detected"

                    state["message"] = (
                        "Position your face clearly "
                        "in front of the camera"
                    )

                # Store one analytics sample per second
                if current_time - last_series_write >= 1.0:

                    state["series"].append({

                        "t": round(elapsed, 1),

                        "ear": round(ear, 3),

                        "mar": round(mar, 3),

                        "perclos": round(perclos, 2),

                        "blink_rate": round(
                            float(len(blink_timestamps)), 1
                        )
                    })

                    state["series"] = state["series"][-600:]

                    last_series_write = current_time

            # =================================================
            # CAMERA OVERLAYS
            # =================================================

            color = (
                (0, 0, 255)
                if drowsy_active
                else (40, 210, 130)
            )

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 50),
                2
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.1f}%",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 50),
                2
            )

            label = (
                "DROWSINESS ALERT"
                if drowsy_active
                else (
                    "FACE DETECTED"
                    if face_found
                    else "NO FACE"
                )
            )

            cv2.putText(
                frame,
                label,
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 82]
            )

            if encoded_ok:

                with lock:
                    latest_frame = encoded.tobytes()

        add_event(
            "System",
            "Monitoring session stopped"
        )

    # ========================================================
    # ERROR HANDLING AND CLEANUP
    # ========================================================

    except Exception as exc:

        with lock:

            state["error"] = str(exc)
            state["status"] = "Error"
            state["message"] = str(exc)

        add_event(
            "Error",
            str(exc)
        )

    finally:

        if cap is not None:
            cap.release()

        if landmarker is not None:
            landmarker.close()

        with lock:

            # Finalize an eye closure if monitoring stopped
            # while the eyes were still classified as closed.
            if closure_start is not None:

                final_closure = max(
                    0.0,
                    time.monotonic() - closure_start
                )

                if drowsy_active:
                    drowsy_accumulated += final_closure

                if final_closure >= MIN_BLINK_DURATION:

                    state["blink_count"] += 1

                    blink_durations.append(
                        final_closure
                    )

                    state["avg_blink_duration"] = (
                        sum(blink_durations)
                        / len(blink_durations)
                    )

                    state["longest_eye_closure"] = max(
                        state["longest_eye_closure"],
                        final_closure
                    )

            state["current_eye_closure"] = 0.0

            state["drowsy_duration"] += (
                drowsy_accumulated
            )

            state["running"] = False

            state["session_elapsed"] = max(
                0.0,
                time.time() - session_start
            )

            if state["status"] != "Error":

                state["status"] = "Stopped"

                state["message"] = (
                    "Monitoring session completed"
                )

        save_session_summary()

        stop_event.clear()


# ============================================================
# 6. FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_detection():

    global worker_thread, latest_frame

    with lock:

        if state["running"]:

            return jsonify({
                "ok": True,
                "message": "Monitoring is already running"
            })

        if worker_thread and worker_thread.is_alive():

            return jsonify({
                "ok": False,
                "message": "Previous monitoring session is still closing"
            }), 409

        stop_event.clear()

        latest_frame = None

        state["running"] = True

        worker_thread = threading.Thread(
            target=detection_worker,
            daemon=True
        )

        worker_thread.start()

    return jsonify({
        "ok": True,
        "message": "Monitoring started"
    })


@app.route("/api/stop", methods=["POST"])
def stop_detection():

    stop_event.set()

    return jsonify({
        "ok": True,
        "message": "Stopping monitoring"
    })


@app.route("/api/status")
def api_status():

    with lock:

        snapshot = {
            k: v
            for k, v in state.items()
            if k not in ("events", "series")
        }

        snapshot["events_count"] = len(
            state["events"]
        )

    return jsonify(snapshot)


@app.route("/api/events")
def api_events():

    with lock:
        return jsonify(state["events"][:30])


@app.route("/api/series")
def api_series():

    with lock:
        return jsonify(state["series"])


@app.route("/api/history")
def api_history():

    path = LOG_DIR / "session_summary.csv"

    rows = []

    if path.exists():

        with path.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            rows = list(
                csv.DictReader(f)
            )

    rows.reverse()

    return jsonify(rows[:50])


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            with lock:

                frame = latest_frame
                active = state["running"]

            if frame is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            elif not active:

                time.sleep(0.2)

            else:

                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# 7. CSV DOWNLOADS
# ============================================================

@app.route("/api/download/events")
def download_events():

    sid = request.args.get("session_id", "")

    if (
        not sid
        or any(
            ch not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in sid
        )
    ):

        abort(400, "Invalid session id")

    path = LOG_DIR / f"events_{sid}.csv"

    if not path.exists():

        abort(
            404,
            "No event log found for this session"
        )

    return send_file(
        path,
        as_attachment=True,
        download_name=f"drowsiness_events_{sid}.csv"
    )


@app.route("/api/download/summary")
def download_summary():

    path = LOG_DIR / "session_summary.csv"

    if not path.exists():

        abort(
            404,
            "No session summary is available yet"
        )

    return send_file(
        path,
        as_attachment=True,
        download_name="session_summary.csv"
    )


# ============================================================
# 8. PDF SESSION REPORT
# ============================================================

@app.route("/api/report/<session_id>")
def download_pdf_report(session_id):

    if (
        not session_id
        or any(
            ch not in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in session_id
        )
    ):

        abort(400)

    summary_path = LOG_DIR / "session_summary.csv"

    if not summary_path.exists():
        abort(404, "No session data available")

    with summary_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        rows = [
            r
            for r in csv.DictReader(f)
            if r["session_id"] == session_id
        ]

    if not rows:
        abort(404, "Session not found")

    row = rows[-1]

    report_path = (
        LOG_DIR / f"report_{session_id}.pdf"
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#64748b")
        )
    )

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    story = [

        Paragraph(
            "Driver Drowsiness Detection",
            styles["Title"]
        ),

        Paragraph(
            "Automated Monitoring Session Report",
            styles["Heading2"]
        ),

        Spacer(1, 5 * mm),

        Paragraph(
            f"<b>Session ID:</b> {session_id}",
            styles["Normal"]
        ),

        Paragraph(
            f"<b>Session date:</b> {row['date']}",
            styles["Normal"]
        ),

        Spacer(1, 5 * mm)
    ]

    table_data = [

        ["Metric", "Recorded value"],

        [
            "Session duration",
            f"{row['duration_seconds']} seconds"
        ],

        [
            "Valid monitoring time",
            f"{row['valid_monitoring_seconds']} seconds"
        ],

        [
            "Drowsiness events",
            row["drowsiness_events"]
        ],

        [
            "Total drowsy duration",
            f"{row['drowsy_duration_seconds']} seconds"
        ],

        [
            "Detected blinks",
            row.get("blink_count", "N/A")
        ],

        [
            "Average blink duration",
            f"{row.get('average_blink_duration_seconds', 'N/A')} seconds"
        ],

        [
            "Longest eye closure",
            f"{row.get('longest_eye_closure_seconds', 'N/A')} seconds"
        ],

        [
            "Possible yawns",
            row["yawns"]
        ],

        [
            "Maximum PERCLOS",
            f"{row['max_perclos_percent']}%"
        ]
    ]

    table = Table(
        table_data,
        colWidths=[75 * mm, 85 * mm]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172554")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd5e1")
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f1f5f9")
                ]
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.extend([

        table,

        Spacer(1, 8 * mm),

        Paragraph(
            "Interpretation",
            styles["Heading2"]
        ),

        Paragraph(
            "This report summarizes measurements collected by an "
            "educational prototype. EAR estimates eye openness, "
            "MAR estimates mouth opening, and PERCLOS represents "
            "the percentage of valid sampled frames classified "
            "as eye-closed during the rolling window. Blink "
            "statistics are based on completed EAR-low closure "
            "cycles. Possible yawns are heuristic detections and "
            "may include speech or other mouth movements. These "
            "values are not a medical assessment or a certified "
            "driver-safety decision.",
            styles["Normal"]
        ),

        Spacer(1, 5 * mm),

        Paragraph(
            "Detection settings",
            styles["Heading2"]
        ),

        Paragraph(
            f"EAR threshold: {EAR_THRESHOLD} | "
            f"Prolonged closure: {DROWSY_SECONDS}s | "
            f"Minimum blink duration: {MIN_BLINK_DURATION}s | "
            f"MAR threshold: {MAR_THRESHOLD} | "
            f"Yawn duration: {YAWN_SECONDS}s | "
            f"PERCLOS window: {PERCLOS_WINDOW:.0f}s",
            styles["Normal"]
        ),

        Spacer(1, 8 * mm),

        Paragraph(
            "Generated locally by DriverSense.",
            styles["SmallNote"]
        )
    ])

    doc.build(story)

    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"drowsiness_report_{session_id}.pdf"
    )


# ============================================================
# 9. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\nDriverSense - Version 3.1")
    print("Open http://127.0.0.1:5000 in your browser\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
