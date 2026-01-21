import time
import cv2

class Camera:
    def __init__(self, device_index=0, resolution=(640, 480), fps=10):
        self.device_index = device_index
        self.resolution = resolution
        self.fps = fps
        self.cap = cv2.VideoCapture(self.device_index)
        # Try to configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.interval = 1.0 / max(1, fps)

    def frames(self):
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible. Ensure /dev/video0 exists and permissions are correct.")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                # brief wait and retry
                time.sleep(0.05)
                continue
            yield frame
            time.sleep(self.interval)

    def __del__(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
