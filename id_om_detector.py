import cv2
import numpy as np
import onnxruntime
import pyclipper
from shapely.geometry import Polygon
from pathlib import Path

class IdOmDetector:
    def __init__(self, onnx_path: str, config_path=None, size=(640, 640), box_thresh=0.7, thresh=0.3):
        self.size = size
        self.onnx_path = onnx_path
        self.box_thresh = box_thresh
        self.thresh = thresh
        self.min_size = 3
        self.max_candidates = 100
        
        # Init Session
        self.session = onnxruntime.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_image(self, img_path_or_array):
        if isinstance(img_path_or_array, (str, Path)):
            img = cv2.imread(str(img_path_or_array))
            if img is None:
                raise ValueError(f"Could not read image: {img_path_or_array}")
        else:
            img = img_path_or_array.copy()

        original_shape = img.shape
        img_resized = cv2.resize(img, self.size)
        
        img_normalized = img_resized.astype('float32')
        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        img_normalized -= RGB_MEAN
        img_normalized /= 255.0
        
        img_array = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img, img_batch, original_shape

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score: continue
        
            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2: continue
            box = np.array(box)
            
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def predict(self, img_path_or_array):
        orig_img, img_batch, orig_shape = self.preprocess_image(img_path_or_array)
        
        preds = self.session.run(None, {self.input_name: img_batch.astype(np.float32)})
        
        if preds[0].ndim == 4:
            pred_map = preds[0][0, 0]
        elif preds[0].ndim == 3:
            pred_map = preds[0][0]
        else:
            pred_map = preds[0]
            
        bitmap = pred_map > self.thresh
        
        dest_height, dest_width = orig_shape[:2]
        boxes, scores = self.boxes_from_bitmap(pred_map, bitmap, dest_width, dest_height)
        
        valid_boxes = [box for box in boxes if not np.all(box == 0)]
        valid_scores = [scores[i] for i, box in enumerate(boxes) if not np.all(box == 0)]
        
        return orig_img, valid_boxes, valid_scores

    def draw_boxes(self, img, boxes):
        out_img = img.copy()
        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(out_img, [box], True, (0, 0, 255), 2)
        return out_img
