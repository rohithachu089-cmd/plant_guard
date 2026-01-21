# Plant Guard (Raspberry Pi + TensorFlow Lite + ESP8266)

A Raspberry Pi app that uses a camera to detect plant disease classes (healthy, powdery, rust) in real time with TensorFlow Lite. It shows a web dashboard with the live camera stream, current prediction probabilities, water level from the ESP, and spray counters. When the model predicts powdery or rust with high confidence, it triggers the respective pump on the ESP8266.

- healthy → no spray
- powdery → spray pump 1
- rust → spray pump 2

The Pi talks to the ESP8266 over simple HTTP endpoints. See the “ESP8266 endpoints” section below.


## Project structure

- `app.py` — Flask web server, camera streaming (MJPEG), inference loop, ESP communication.
- `camera.py` — OpenCV camera capture.
- `inference.py` — TensorFlow Lite inference wrapper.
- `train.py` — Transfer learning with MobileNetV2, exports int8 quantized TFLite model.
- `config.py` — Configuration (ESP IP, thresholds, ports, model path).
- `templates/index.html` — Bootstrap-based UI.
- `requirements.txt` — Python packages.
- `models/` — Put exported TFLite model and labels here.


## Hardware and OS notes

- Raspberry Pi OS Bookworm/Bullseye.
- Camera:
  - USB webcam works out of the box with OpenCV.
  - For the Raspberry Pi Camera Module, enable the V4L2 compatibility layer if needed:
    ```bash
    sudo apt-get update
    sudo apt-get install -y v4l2loopback-utils
    sudo modprobe bcm2835-v4l2
    # To load on boot:
    echo 'bcm2835-v4l2' | sudo tee -a /etc/modules
    ```
  - Verify with `v4l2-ctl --list-devices` and check `/dev/video0` exists.


## Install on Raspberry Pi

1) System packages and Python venv

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv libatlas-base-dev libjpeg-dev libopenjp2-7

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2) Install Python packages

By default, this project prefers lightweight `tflite-runtime` on Raspberry Pi. Install one of the following:

- Option A (recommended, Pi aarch64/armv7):
```bash
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime==2.14.0
```

- Option B (heavier, but includes everything):
```bash
pip install tensorflow==2.14.0
```

Then install the rest:
```bash
pip install -r requirements.txt --no-deps
```

Note: we pass `--no-deps` because you already installed either `tflite-runtime` or `tensorflow`. The rest of packages are light.


## Prepare your dataset

Organize your images as:
```
/your/dataset/
  healthy/
    img1.jpg
    ...
  powdery/
    img2.jpg
    ...
  rust/
    img3.jpg
    ...
```
Images will be resized to 224x224 and normalized to [0,1].


## Train and export TensorFlow Lite model

Run training on the Pi or (faster) on your PC. Copy the project and dataset over.

```bash
source .venv/bin/activate
python train.py --data /path/to/your/dataset --epochs 10 --fine_tune_epochs 5
```

Artifacts written to `models/`:
- `plant_disease_int8.tflite`
- `labels.txt` (order must match model outputs)

You can adjust thresholds in `config.py`:
- `PREDICTION_SMOOTHING` (temporal averaging)
- `PREDICTION_THRESHOLD` (default 0.78)
- `AUTO_SPRAY_COOLDOWN_SEC` (e.g., 30s)


## Configure ESP endpoint in config

Edit `config.py` and set:
```python
ESP_BASE_URL = "http://<esp-ip-address>"
```
Make sure your ESP serves:
- `GET /status` → `{ "water_level": 0-100, "pump1_count": int, "pump2_count": int }`
- `GET /spray?pump=1` → triggers pump 1 (powdery)
- `GET /spray?pump=2` → triggers pump 2 (rust)

An example ESP8266 sketch is in `esp_endpoints_example.ino`.


## Run the app

```bash
source .venv/bin/activate
python app.py
```
Open the dashboard at:
```
http://<raspberry-pi-ip>:8000
```

- Toggle Auto mode with the button at top-right.
- Manual spray buttons are under “Manual Control”.
- Water level and spray counts are polled from the ESP every 2 seconds.


## Run as a service (optional)

Create a systemd service at `/etc/systemd/system/plant-guard.service`:

```ini
[Unit]
Description=Plant Guard
After=network.target

[Service]
WorkingDirectory=/home/pi/plant_guard
ExecStart=/home/pi/plant_guard/.venv/bin/python app.py
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Then run:
```bash
sudo systemctl daemon-reload
sudo systemctl enable plant-guard
sudo systemctl start plant-guard
```


## ESP8266 endpoints

Add two HTTP endpoints to your existing firmware using `ESP8266WebServer`:
- `GET /status` should return JSON with the water level and pump counters.
- `GET /spray?pump=1|2` should trigger your existing pump logic.

See `esp_endpoints_example.ino` in this folder. Adjust pins and logic as needed.


## Tips for accuracy

- Ensure your dataset is balanced across the three classes.
- Use clear, well-lit images; avoid heavy motion blur.
- Increase `--epochs` and `--fine_tune_epochs` if underfitting; consider unfreezing more of MobileNetV2 layers.
- Consider data augmentation in `train.py` (RandomFlip/Rotation/Zoom) if the model overfits.
- Keep the camera at a consistent distance and angle.


## Troubleshooting

- No camera: ensure `/dev/video0` exists and try `sudo modprobe bcm2835-v4l2` for Pi Camera Module.
- Performance: reduce `CAMERA_RESOLUTION` and increase `INFERENCE_FPS` carefully.
- ESP unreachable: verify Pi and ESP on same network; test `curl http://<esp-ip>/status`.
