# Configuration for Plant Guard Raspberry Pi app

# === ESP8266/ESP32 Communication ===
# Set the base URL of your ESP (ensure it serves simple HTTP endpoints as documented in README)
ESP_BASE_URL = "http://192.168.1.50"  # change to your ESP IP or mDNS name
ESP_STATUS_PATH = "/status"          # GET returns JSON {"water_level": 0-100, "pump1_count": int, "pump2_count": int}
ESP_SPRAY_PATH = "/spray"            # GET/POST with param pump=1|2 triggers pump
ESP_TIMEOUT_SEC = 3

# === Model & Inference ===
MODEL_DIR = "models"
TFLITE_MODEL_PATH = f"{MODEL_DIR}/plant_disease_int8.tflite"
LABELS_PATH = f"{MODEL_DIR}/labels.txt"

# MobileNetV2 default input size
MODEL_INPUT_SIZE = (224, 224)  # (H, W)
PREDICTION_SMOOTHING = 5       # number of recent predictions to smooth
PREDICTION_THRESHOLD = 0.78    # min probability to accept decision
INFERENCE_FPS = 3              # run N inferences per second
AUTO_SPRAY_COOLDOWN_SEC = 30   # per-class cooldown to avoid repeated sprays

# === Camera ===
CAMERA_FPS = 10
CAMERA_RESOLUTION = (640, 480)

# === UI ===
APP_HOST = "0.0.0.0"
APP_PORT = 8000
DEBUG = False

# === Safety ===
# Set to False to prevent any automatic spraying (testing mode)
AUTO_ENABLED_DEFAULT = True
