from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from database import get_db, DATA_DIR
from dependencies import get_current_user
import models
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from inference_service import letterbox, DEFAULT_BASE_CHARSET
from inference_kie import get_parseq_transform, parseq_decode

router = APIRouter()

_session_cache = {}

def get_ort_session(onnx_path: str):
    if onnx_path not in _session_cache:
        _session_cache[onnx_path] = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return _session_cache[onnx_path]

def run_yolo_live_predict(onnx_path: str, img_path: str, project: models.Project, conf_thres: float = 0.25, iou_thres: float = 0.45):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    orig_h, orig_w = img.shape[:2]
    
    session = get_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Determine size dynamically
    imgsz = 640
    if project.model_img_h and int(project.model_img_h) > 0:
        imgsz = int(project.model_img_h)
    elif len(input_shape) == 4:
        h_in = input_shape[2]
        if isinstance(h_in, int) and h_in > 0:
            imgsz = h_in
            
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, (scale_w, scale_h), (pad_w, pad_h) = letterbox(img_rgb, new_shape=(imgsz, imgsz))
    scale = scale_w
    
    img_tensor = img_resized.transpose((2, 0, 1))
    img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    outputs = session.run(None, {input_name: img_tensor})
    preds = outputs[0]
    
    results = []
    
    if preds.ndim == 3 and preds.shape[-1] == 6:
        # YOLOv10/v8 NMS format: (1, 300, 6)
        for det in preds[0]:
            x_min, y_min, x_max, y_max, conf, cls_id = det
            if conf >= conf_thres:
                x1_orig = (x_min - pad_w) / scale
                y1_orig = (y_min - pad_h) / scale
                x2_orig = (x_max - pad_w) / scale
                y2_orig = (y_max - pad_h) / scale
                
                cx_norm = ((x1_orig + x2_orig) / 2.0) / orig_w
                cy_norm = ((y1_orig + y2_orig) / 2.0) / orig_h
                w_norm = abs(x2_orig - x1_orig) / orig_w
                h_norm = abs(y2_orig - y1_orig) / orig_h
                
                cx_norm = max(0.0, min(1.0, cx_norm))
                cy_norm = max(0.0, min(1.0, cy_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                results.append({
                    "class_code": int(cls_id),
                    "coordinates": f"{cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                })
    else:
        # Standard YOLOv8 format: (1, 4 + C, 8400)
        preds = np.squeeze(preds)
        if len(preds.shape) == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.transpose()
            
        if preds.shape[1] >= 5:
            boxes = preds[:, :4]
            scores = preds[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            mask = confidences > conf_thres
            boxes = boxes[mask]
            confidences = confidences[mask]
            class_ids = class_ids[mask]
            
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
            x1 = x - w / 2
            y1 = y - h / 2
            
            boxes_nms = np.column_stack((x1, y1, w, h)).tolist()
            indices = cv2.dnn.NMSBoxes(boxes_nms, confidences.tolist(), conf_thres, iou_thres)
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes_nms[idx]
                    bx_orig = (bx - pad_w) / scale
                    by_orig = (by - pad_h) / scale
                    bw_orig = bw / scale
                    bh_orig = bh / scale
                    
                    cx_orig = bx_orig + bw_orig / 2.0
                    cy_orig = by_orig + bh_orig / 2.0
                    
                    cx_norm = cx_orig / orig_w
                    cy_norm = cy_orig / orig_h
                    w_norm = bw_orig / orig_w
                    h_norm = bh_orig / orig_h
                    
                    cx_norm = max(0.0, min(1.0, cx_norm))
                    cy_norm = max(0.0, min(1.0, cy_norm))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    results.append({
                        "class_code": int(class_ids[idx]),
                        "coordinates": f"{cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                    })
                    
    return {"boxes": results}


def run_yolo_obb_live_predict(onnx_path: str, img_path: str, project: models.Project, conf_thres: float = 0.25, iou_thres: float = 0.45):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    orig_h, orig_w = img.shape[:2]
    
    session = get_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Determine size dynamically
    imgsz = None
    if project.model_img_h and int(project.model_img_h) > 0:
        imgsz = int(project.model_img_h)
    else:
        h_in = input_shape[2] if len(input_shape) == 4 else None
        if isinstance(h_in, int) and h_in > 0:
            imgsz = h_in
        else:
            # Dynamic axes — auto-probe common YOLO sizes to find best confidence
            best_imgsz = 640
            best_conf = -1.0
            probe_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for candidate in [640, 1024, 512, 800, 1280]:
                try:
                    probe_resized = cv2.resize(probe_img, (candidate, candidate))
                    t = probe_resized.transpose((2, 0, 1))
                    t = np.ascontiguousarray(t, dtype=np.float32) / 255.0
                    t = np.expand_dims(t, axis=0)
                    out = session.run(None, {input_name: t})[0]
                    if out.ndim == 3 and out.shape[1] < 1000 and out.shape[-1] >= 5:
                        max_conf = float(np.max(out[0, :, 4]))
                    elif out.ndim == 3:
                        arr = out[0]
                        if arr.shape[0] < arr.shape[1]:
                            max_conf = float(np.max(arr[4:, :]))
                        else:
                            max_conf = float(np.max(arr[:, 4:]))
                    else:
                        max_conf = 0.0
                    if max_conf > best_conf:
                        best_conf = max_conf
                        best_imgsz = candidate
                except Exception:
                    pass
            imgsz = best_imgsz

    # Letterbox — identical to DataToolBox letterbox_image_cv
    h0, w0 = orig_h, orig_w
    r = min(imgsz / w0, imgsz / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw = (imgsz - nw) / 2.0
    dh = (imgsz - nh) / 2.0
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_left   = int(round(dw - 0.1))
    pad_top    = int(round(dh - 0.1))
    pad_right  = imgsz - nw - pad_left
    pad_bottom = imgsz - nh - pad_top
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR→RGB, HWC→NCHW, /255
    blob = img_padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]

    preds = session.run(None, {input_name: blob})[0]

    results = []

    if preds.ndim == 3 and preds.shape[1] < 1000 and preds.shape[-1] == 7:
        # NMS-embedded format: [1, N, 7] → [cx, cy, w, h, conf, cls_id, angle]
        for row in preds[0]:
            cx, cy, w, h, conf, class_id, angle = row
            if conf < conf_thres:
                continue
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            vec1 = np.array([ w/2 * cos_a,  w/2 * sin_a])
            vec2 = np.array([-h/2 * sin_a,  h/2 * cos_a])
            ctr  = np.array([cx, cy])
            poly_padded = np.array([ctr+vec1+vec2, ctr+vec1-vec2, ctr-vec1-vec2, ctr-vec1+vec2])
            poly_orig = poly_padded.copy()
            poly_orig[:, 0] = (poly_padded[:, 0] - pad_left) / r
            poly_orig[:, 1] = (poly_padded[:, 1] - pad_top)  / r
            poly_norm = np.clip(poly_orig / np.array([orig_w, orig_h]), 0.0, 1.0)
            coords_str = " ".join([f"{c:.6f}" for c in poly_norm.flatten()])
            results.append({"class_code": int(class_id), "coordinates": coords_str})
        return {"boxes": results}

    # Raw output: [1, 5+C, 8400] — mirrors inference_yolov8_obb_pure_onnx.py postprocess()
    import math as _math
    raw = preds[0].T  # → (8400, 5+C)
    num_classes = raw.shape[1] - 5
    boxes_f  = raw[:, :4]
    scores_f = raw[:, 4:4+num_classes]
    angles_f = raw[:, -1]

    max_scores = np.max(scores_f, axis=1)
    class_ids  = np.argmax(scores_f, axis=1)

    # Pre-filter at 0.01, NMS, then apply real conf_thres (matches reference exactly)
    keep = max_scores >= 0.01
    boxes_f  = boxes_f[keep]
    max_scores = max_scores[keep]
    class_ids  = class_ids[keep]
    angles_f = angles_f[keep]

    if len(boxes_f) == 0:
        return {"boxes": []}

    rotated_nms = [
        ((float(cx), float(cy)), (float(w), float(h)), _math.degrees(float(a)))
        for (cx, cy, w, h), a in zip(boxes_f, angles_f)
    ]
    idxs = cv2.dnn.NMSBoxesRotated(rotated_nms, max_scores.tolist(), 0.01, iou_thres)

    if len(idxs) > 0:
        for idx in idxs.flatten():
            conf = float(max_scores[idx])
            if conf < conf_thres:
                continue
            gain = min(imgsz / orig_w, imgsz / orig_h)
            px = int(round((imgsz - round(orig_w * gain)) / 2.0 - 0.1))
            py = int(round((imgsz - round(orig_h * gain)) / 2.0 - 0.1))
            cx, cy, w, h = boxes_f[idx]
            cx = (cx - px) / gain
            cy = (cy - py) / gain
            w  = w  / gain
            h  = h  / gain
            angle = float(angles_f[idx])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            vec1 = np.array([ w/2 * cos_a,  w/2 * sin_a])
            vec2 = np.array([-h/2 * sin_a,  h/2 * cos_a])
            ctr  = np.array([cx, cy])
            poly_orig = np.array([ctr+vec1+vec2, ctr+vec1-vec2, ctr-vec1-vec2, ctr-vec1+vec2])
            poly_norm = np.clip(poly_orig / np.array([orig_w, orig_h]), 0.0, 1.0)
            coords_str = " ".join([f"{c:.6f}" for c in poly_norm.flatten()])
            results.append({"class_code": int(class_ids[idx]), "coordinates": coords_str})

    return {"boxes": results}


def run_ocr_live_predict(onnx_path: str, img_path: str, project: models.Project, conf_thresh: float = 0.0):
    session = get_ort_session(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    h, w = 48, 256
    
    # Priority 1: Project settings
    if project.model_img_h and int(project.model_img_h) > 0:
        h = int(project.model_img_h)
    elif len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
        h = input_shape[2]
        
    if project.model_img_w and int(project.model_img_w) > 0:
        w = int(project.model_img_w)
    elif len(input_shape) == 4 and isinstance(input_shape[3], int) and input_shape[3] > 0:
        w = input_shape[3]
            
    img = Image.open(img_path)
    image_gray = img.convert("L")
    image_bw_rgb = image_gray.convert("RGB")
    
    transform = get_parseq_transform((h, w))
    tensor = transform(image_bw_rgb)
    
    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0]
    
    charset = project.ocr_charset if project.ocr_charset else DEFAULT_BASE_CHARSET
    pred_text, conf = parseq_decode(logits, charset)
    
    if conf < conf_thresh:
        pred_text = ""
        
    return {"text": pred_text}


@router.post("/{project_id}/predict-live")
def predict_live(
    project_id: int,
    img_name: str = Body(..., embed=True),
    conf_thresh: float = Body(0.25, embed=True),
    iou_thresh: float = Body(0.45, embed=True),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if not project.dbnet_model_path:
        raise HTTPException(status_code=400, detail="No model path defined for this project")
        
    onnx_path = os.path.normpath(project.dbnet_model_path)
    if not os.path.exists(onnx_path):
        raise HTTPException(status_code=400, detail=f"ONNX model file not found at: {onnx_path}")
        
    img_path = os.path.join(DATA_DIR, "tmp_uploads", str(project_id), "local_workspace", img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(DATA_DIR, str(project_id), img_name)
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Image not found")
            
    try:
        # Check if the model is DBNet
        is_dbnet = "dbnet" in os.path.basename(onnx_path).lower()
        if not is_dbnet:
            # Check by inspecting the output shape
            temp_session = get_ort_session(onnx_path)
            outputs = temp_session.get_outputs()
            if len(outputs) > 0:
                shape = outputs[0].shape
                if len(shape) == 4 and shape[1] in [1, 2]:
                    is_dbnet = True
                    
        if is_dbnet:
            from inference_kie import get_id_om_detector
            detector = get_id_om_detector(onnx_path=onnx_path)
            detector.box_thresh = conf_thresh
            _, boxes, _ = detector.predict(img_path)
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            img_h, img_w = img.shape[:2]
            
            # Fetch first class code
            first_class = db.query(models.Class).filter(models.Class.project_id == project_id).order_by(models.Class.code).first()
            class_code = first_class.code if first_class else 0
            
            results = []
            for box in boxes:
                box = np.array(box) # shape (4, 2)
                if project.type == "Yolo OBB":
                    # Normalized corners: x1 y1 x2 y2 x3 y3 x4 y4
                    pts_norm = box.astype(np.float32)
                    pts_norm[:, 0] /= img_w
                    pts_norm[:, 1] /= img_h
                    pts_norm = np.clip(pts_norm, 0.0, 1.0)
                    coordsStr = " ".join([f"{c:.6f}" for c in pts_norm.flatten()])
                else:
                    # Standard Yolo format: cx cy w h
                    xmin = float(np.min(box[:, 0])) / img_w
                    ymin = float(np.min(box[:, 1])) / img_h
                    xmax = float(np.max(box[:, 0])) / img_w
                    ymax = float(np.max(box[:, 1])) / img_h
                    w = max(0.001, xmax - xmin)
                    h = max(0.001, ymax - ymin)
                    cx = xmin + w / 2
                    cy = ymin + h / 2
                    
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    coordsStr = f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    
                results.append({
                    "class_code": class_code,
                    "coordinates": coordsStr
                })
            return {"boxes": results}
            
        # Non-DBNet routing
        if project.type == "Yolo":
            return run_yolo_live_predict(onnx_path, img_path, project, conf_thres=conf_thresh, iou_thres=iou_thresh)
        elif project.type == "Yolo OBB":
            return run_yolo_obb_live_predict(onnx_path, img_path, project, conf_thres=conf_thresh, iou_thres=iou_thresh)
        elif project.type == "Ocr":
            return run_ocr_live_predict(onnx_path, img_path, project, conf_thresh=conf_thresh)
        else:
            raise HTTPException(status_code=400, detail=f"Live prediction not supported for project type: {project.type}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



from fastapi import Body
import cv2
import numpy as np
from PIL import Image
from database import DATA_DIR
from inference_kie import get_id_om_detector, get_parseq_om_recognizer

@router.post("/{project_id}/kie/detect-boxes")
def kie_detect_boxes(
    project_id: int,
    img_name: str = Body(..., embed=True),
    box_thresh: float = Body(0.7, embed=True),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")

    img_path = os.path.join(DATA_DIR, "tmp_uploads", str(project_id), "local_workspace", img_name)
    if not os.path.exists(img_path):
        # Fallback to permanent directory
        img_path = os.path.join(DATA_DIR, str(project_id), img_name)
        if not os.path.exists(img_path):
            raise HTTPException(404, "Image not found: " + img_path)

    try:
        detector = get_id_om_detector(onnx_path=project.dbnet_model_path)
        detector.box_thresh = box_thresh
        # predict returns: orig_img, valid_boxes, valid_scores
        _, boxes, _ = detector.predict(img_path)
        
        img_h, img_w = cv2.imread(img_path).shape[:2]
        
        results = []
        for box in boxes:
            box = np.array(box)
            xmin = float(np.min(box[:, 0])) / img_w
            ymin = float(np.min(box[:, 1])) / img_h
            xmax = float(np.max(box[:, 0])) / img_w
            ymax = float(np.max(box[:, 1])) / img_h
            w = max(0.001, xmax - xmin)
            h = max(0.001, ymax - ymin)
            cx = xmin + w / 2
            cy = ymin + h / 2
            coordsStr = f"{cx} {cy} {w} {h}"
            results.append(coordsStr)
            
        return {"boxes": results}
    except Exception as e:
        print(f"Error detect: {e}")
        raise HTTPException(500, str(e))

@router.post("/{project_id}/kie/read-text")
def kie_read_text(
    project_id: int,
    img_name: str = Body(..., embed=True),
    boxes: list = Body(..., embed=True),
    min_confidence: float = Body(0.0, embed=True),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")

    img_path = os.path.join(DATA_DIR, "tmp_uploads", str(project_id), "local_workspace", img_name)
    if not os.path.exists(img_path):
        # Fallback to permanent directory
        img_path = os.path.join(DATA_DIR, str(project_id), img_name)
        if not os.path.exists(img_path):
            raise HTTPException(404, "Image not found: " + img_path)

    try:
        charset = project.ocr_charset if project.ocr_charset else None
        recognizer = get_parseq_om_recognizer(charset=charset)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        results = []
        for box_str in boxes:
            try:
                cx, cy, w, h = map(float, box_str.split())
                xmin = int((cx - w/2) * img_w)
                ymin = int((cy - h/2) * img_h)
                xmax = int((cx + w/2) * img_w)
                ymax = int((cy + h/2) * img_h)
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)
                
                crop = img[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    results.append("")
                    continue
                    
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                text, conf = recognizer.recognize(crop_pil)
                if conf >= min_confidence:
                    results.append(text)
                else:
                    results.append("")
            except Exception as e:
                print(f"Error reading box {box_str}: {e}")
                results.append("")
                
        return {"texts": results}
    except Exception as e:
        print(f"Error read text: {e}")
        raise HTTPException(500, str(e))
