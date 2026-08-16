import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from typing import List, Tuple
from inference_service import DEFAULT_BASE_CHARSET

from id_om_detector import IdOmDetector

def get_parseq_transform(img_size):
    def transform(image: Image.Image) -> np.ndarray:
        w, h = img_size[1], img_size[0]
        img = image.convert("RGB").resize((w, h), Image.BICUBIC)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        img_arr = (img_arr - 0.5) / 0.5
        return np.expand_dims(img_arr, axis=0)
    return transform

def parseq_decode(logits: np.ndarray, charset: str) -> tuple[str, float]:
    eos_id = 0
    itos = ['[E]'] + list(charset) + ['[B]', '[P]']
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)
    pred = preds[0]
    prob = probs[0]
    try:
        eos_idx = list(pred).index(eos_id)
        pred = pred[:eos_idx]
    except ValueError:
        pass
    text = ""
    confidences = []
    for i, idx in enumerate(pred):
        if idx < len(itos) and idx != 0 and itos[idx] not in ('[B]', '[P]'):
            text += itos[idx]
            confidences.append(prob[i, idx])
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, mean_conf

class ParseqOmRecognizer:
    def __init__(self, onnx_path: str, charset: str = DEFAULT_BASE_CHARSET):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Le modèle a été entrainé avec du size [48,256]
        self.img_size = (48, 256)
        self.transform = get_parseq_transform(self.img_size)
        self.charset = charset

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        image_gray = image.convert("L")
        image_bw_rgb = image_gray.convert("RGB")
        tensor = self.transform(image_bw_rgb)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]
        return parseq_decode(logits, self.charset)

# Singleton instances
_id_om_detectors = {}
_parseq_om_recognizer = None

def get_id_om_detector(onnx_path: str = None):
    global _id_om_detectors
    if not onnx_path:
        onnx_path = os.path.join("data", "dbnet_id_om.onnx")
    
    onnx_path = os.path.normpath(onnx_path)
    
    if onnx_path not in _id_om_detectors:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        _id_om_detectors[onnx_path] = IdOmDetector(onnx_path=onnx_path)
    return _id_om_detectors[onnx_path]

def get_parseq_om_recognizer(charset: str = None):
    global _parseq_om_recognizer
    if _parseq_om_recognizer is None:
        onnx_path = os.path.join("data", "parseq_om.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        _parseq_om_recognizer = ParseqOmRecognizer(onnx_path=onnx_path, charset=charset or DEFAULT_BASE_CHARSET)
    return _parseq_om_recognizer
