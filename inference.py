import time
import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception:
        Interpreter = None

class TFLitePlantClassifier:
    def __init__(self, model_path: str, labels: list[str], input_size=(224, 224)):
        self.labels = labels
        self.input_h, self.input_w = input_size
        if Interpreter is None:
            self.interpreter = None
            print("Warning: TFLite Interpreter not available. Classification will be disabled.")
            return
        
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_is_quant = self.input_details[0]['dtype'] == np.uint8
            self.input_scale, self.input_zero = self._get_scale_zero(self.input_details[0])
            self.output_is_quant = self.output_details[0]['dtype'] == np.uint8
            self.output_scale, self.output_zero = self._get_scale_zero(self.output_details[0])
        except Exception as e:
            self.interpreter = None
            print(f"Warning: Failed to load model: {e}")

    def _get_scale_zero(self, detail):
        quant = detail.get('quantization', None) or detail.get('quantization_parameters', None)
        if quant and isinstance(quant, tuple):
            scale, zero = quant
            if isinstance(scale, (list, np.ndarray)):
                scale = scale[0] if len(scale) else 1.0
            if isinstance(zero, (list, np.ndarray)):
                zero = zero[0] if len(zero) else 0
            return float(scale or 1.0), int(zero or 0)
        elif quant and 'scales' in quant:
            scales = quant['scales']
            zero_points = quant['zero_points']
            scale = float(scales[0] if len(scales) else 1.0)
            zero = int(zero_points[0] if len(zero_points) else 0)
            return scale, zero
        return 1.0, 0

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.input_is_quant:
            # Assume input expects 0..255 uint8
            tensor = img.astype(np.uint8)
            if self.input_scale != 1.0 or self.input_zero != 0:
                # map float [0,1] to quant domain if required (rare for uint8 inputs)
                tensor = (img.astype(np.float32) / 255.0 / self.input_scale + self.input_zero).round().astype(np.uint8)
        else:
            # float32 0..1
            tensor = img.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, frame_bgr: np.ndarray):
        if self.interpreter is None:
            return "healthy", {l: 0.0 for l in self.labels}
        input_tensor = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        if self.output_is_quant:
            output = (output.astype(np.float32) - self.output_zero) * self.output_scale
        probs = output[0]
        # Softmax normalization safeguard
        if probs.ndim == 1:
            # Some models may output logits
            exps = np.exp(probs - np.max(probs))
            probs = exps / np.sum(exps)
        probs = probs.astype(np.float32)
        label_probs = {self.labels[i]: float(probs[i]) for i in range(min(len(self.labels), len(probs)))}
        best_idx = int(np.argmax(probs))
        best_label = self.labels[best_idx]
        return best_label, label_probs
