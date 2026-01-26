import os
import time
import threading
import queue
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import requests
# Removed redundant/incorrectly placed initialization

from config import (
    ESP_BASE_URL,
    ESP_STATUS_PATH,
    ESP_SPRAY_PATH,
    ESP_MOVE_PATH,
    ESP_TIMEOUT_SEC,
    MODEL_INPUT_SIZE,
    TFLITE_MODEL_PATH,
    LABELS_PATH,
    PREDICTION_SMOOTHING,
    PREDICTION_THRESHOLD,
    INFERENCE_FPS,
    AUTO_SPRAY_COOLDOWN_SEC,
    SCAN_INTERVAL_SEC,
    SCAN_DURATION_SEC,
    CAMERA_FPS,
    CAMERA_RESOLUTION,
    APP_HOST,
    APP_PORT,
    DEBUG,
    AUTO_ENABLED_DEFAULT,
)
from inference import TFLitePlantClassifier
from camera import Camera


app = Flask(__name__)

# Shared state
state_lock = threading.Lock()
auto_enabled = AUTO_ENABLED_DEFAULT
scan_active = False # New state for automated routine
bot_phase = "stopped" # "moving" or "scanning" or "stopped"
last_spray_time = {"powdery": 0, "rust": 0}
spray_counts_local = {"pump1": 0, "pump2": 0}
last_prediction = {"label": "", "probs": {}, "timestamp": 0}
water_level_cache = 0

# Frame queues
frame_queue = queue.Queue(maxsize=2)
prediction_queue = queue.Queue(maxsize=1)


# Initialize components
try:
    if os.path.exists(LABELS_PATH):
        labels = [line.strip() for line in open(LABELS_PATH, "r", encoding="utf-8").read().splitlines() if line.strip()]
    else:
        labels = []
except Exception:
    labels = []
if not labels:
    labels = ["healthy", "powdery", "rust"]
try:
    classifier = TFLitePlantClassifier(TFLITE_MODEL_PATH, labels, input_size=MODEL_INPUT_SIZE)
except Exception as e:
    print(f"Warning: Classifier could not be initialized: {e}")
    classifier = None
camera = Camera(device_index=0, resolution=CAMERA_RESOLUTION, fps=CAMERA_FPS)


def fetch_esp_status():
    global water_level_cache
    url = f"{ESP_BASE_URL}{ESP_STATUS_PATH}"
    try:
        r = requests.get(url, timeout=ESP_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        with state_lock:
            water_level_cache = int(data.get("water_level", water_level_cache))
            # use ESP counters as source of truth if present
            if "pump1_count" in data:
                spray_counts_local["pump1"] = int(data["pump1_count"])
            if "pump2_count" in data:
                spray_counts_local["pump2"] = int(data["pump2_count"])
        return data
    except Exception:
        # Keep last known values on error
        return {
            "water_level": water_level_cache,
            "pump1_count": spray_counts_local["pump1"],
            "pump2_count": spray_counts_local["pump2"],
        }


def trigger_spray(pump: int):
    assert pump in (1, 2)
    url = f"{ESP_BASE_URL}{ESP_SPRAY_PATH}"
    try:
        r = requests.get(url, params={"pump": pump}, timeout=ESP_TIMEOUT_SEC)
        r.raise_for_status()
    except Exception:
        pass
    # Update local count optimistically
    with state_lock:
        key = "pump1" if pump == 1 else "pump2"
        spray_counts_local[key] += 1


def trigger_move(cmd: str):
    assert cmd in ("start", "stop")
    url = f"{ESP_BASE_URL}{ESP_MOVE_PATH}"
    try:
        r = requests.get(url, params={"cmd": cmd}, timeout=ESP_TIMEOUT_SEC)
        r.raise_for_status()
    except Exception as e:
        print(f"Error triggering move {cmd}: {e}")

def sequencer_loop():
    global bot_phase
    while True:
        with state_lock:
            active = scan_active
        if not active:
            with state_lock:
                if bot_phase != "stopped":
                    trigger_move("stop")
                    bot_phase = "stopped"
            time.sleep(1)
            continue
        
        # Move phase
        with state_lock:
            bot_phase = "moving"
        trigger_move("start")
        time.sleep(SCAN_INTERVAL_SEC)
        
        # Stop & Scan phase
        with state_lock:
            bot_phase = "scanning"
        trigger_move("stop")
        time.sleep(SCAN_DURATION_SEC)


def mjpeg_generator():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def camera_loop():
    for frame in camera.frames():
        # Put latest frame, drop older
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)


def inference_loop():
    global last_prediction
    interval = 1.0 / max(1, INFERENCE_FPS)
    prev_preds = []  # smoothing buffer
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # Run inference
        pred_label, probs = classifier.predict(frame)
        timestamp = time.time()
        prev_preds.append((pred_label, probs))
        if len(prev_preds) > PREDICTION_SMOOTHING:
            prev_preds.pop(0)
        # Smooth by averaging probabilities
        avg_probs = {k: float(np.mean([p[1].get(k, 0.0) for p in prev_preds])) for k in classifier.labels}
        best_label = max(avg_probs.items(), key=lambda x: x[1])[0]
        best_prob = avg_probs[best_label]

        with state_lock:
            last_prediction = {"label": best_label, "probs": avg_probs, "timestamp": timestamp}

        # Draw overlay for stream
        overlay_frame = frame.copy()
        text = f"{best_label} {best_prob*100:.1f}%"
        cv2.putText(overlay_frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if best_label=="healthy" else (0, 165, 255), 2, cv2.LINE_AA)
        # Send overlay frame back to stream queue (optional enhancement)
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(overlay_frame)

        # Auto spray logic (only when scanning or manually auto-enabled)
        with state_lock:
            do_auto = auto_enabled
            current_phase = bot_phase
            is_scanning = (current_phase == "scanning") or (not scan_active) # allow auto if routine is off
        
        if do_auto and is_scanning and best_prob >= PREDICTION_THRESHOLD:
            now = time.time()
            if best_label == "powdery" and now - last_spray_time["powdery"] >= AUTO_SPRAY_COOLDOWN_SEC:
                trigger_spray(1)
                last_spray_time["powdery"] = now
            elif best_label == "rust" and now - last_spray_time["rust"] >= AUTO_SPRAY_COOLDOWN_SEC:
                trigger_spray(2)
                last_spray_time["rust"] = now

        time.sleep(interval)


@app.route('/')
def index():
    status = fetch_esp_status()
    with state_lock:
        lp = last_prediction.copy()
        auto = auto_enabled
        counts = spray_counts_local.copy()
    return render_template('index.html', status=status, last_prediction=lp, auto_enabled=auto, counts=counts, labels=classifier.labels if classifier else ["healthy", "powdery", "rust"])


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    status = fetch_esp_status()
    with state_lock:
        lp = last_prediction.copy()
        auto = auto_enabled
        counts = spray_counts_local.copy()
    return jsonify({
        "water_level": status.get("water_level", 0),
        "pump1_count": status.get("pump1_count", counts["pump1"]),
        "pump2_count": status.get("pump2_count", counts["pump2"]),
        "last_prediction": lp,
        "auto_enabled": auto,
        "scan_active": scan_active,
        "bot_phase": bot_phase
    })

@app.route('/api/toggle_scan', methods=['POST'])
def api_toggle_scan():
    global scan_active
    with state_lock:
        scan_active = not scan_active
        val = scan_active
    return jsonify({"scan_active": val})


@app.route('/api/toggle_auto', methods=['POST'])
def api_toggle_auto():
    global auto_enabled
    with state_lock:
        auto_enabled = not auto_enabled
        val = auto_enabled
    return jsonify({"auto_enabled": val})


@app.route('/api/spray', methods=['POST'])
def api_spray():
    data = request.get_json(silent=True) or request.form
    try:
        pump = int(data.get('pump'))
    except Exception:
        return jsonify({"ok": False, "error": "pump must be 1 or 2"}), 400
    if pump not in (1, 2):
        return jsonify({"ok": False, "error": "pump must be 1 or 2"}), 400
    trigger_spray(pump)
    return jsonify({"ok": True})


def main():
    # Start camera, inference and sequencer threads
    t_cam = threading.Thread(target=camera_loop, daemon=True)
    t_inf = threading.Thread(target=inference_loop, daemon=True)
    t_seq = threading.Thread(target=sequencer_loop, daemon=True)
    t_cam.start()
    t_inf.start()
    t_seq.start()

    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG, threaded=True)


if __name__ == '__main__':
    main()
