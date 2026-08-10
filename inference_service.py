import os
import json
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from sqlalchemy.orm import Session
from database import SessionLocal, DATA_DIR
from models import Project, YoloLabel, YoloPrelabel, OcrLabel, OcrPrelabel, SkippedImage
from pathlib import Path

# Default charset for PARSeq OCR
DEFAULT_BASE_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZeopsué '-./"



def get_unannotated_and_skipped_images(db: Session, project_id: int, project_type: str):
    """
    Returns a list of image names (strings) that need auto-annotation.
    This includes images in SkippedImage and images in the project directory
    that have no labels or prelabels.
    """
    # 1. Get all images from project directory
    project_dir = DATA_DIR / str(project_id)
    if not project_dir.exists():
        return []
    
    all_images = set(
        f for f in os.listdir(project_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    )
    
    # 2. Get skipped images
    skipped_records = db.query(SkippedImage).filter(SkippedImage.project_id == project_id).all()
    skipped_names = set(record.img_name for record in skipped_records)
    
    # 3. Get labeled and prelabeled images depending on type
    labeled_names = set()
    prelabeled_names = set()
    
    if project_type == "Yolo" or project_type == "Yolo OBB":
        labeled = db.query(YoloLabel.img_name).filter(YoloLabel.project_id == project_id).all()
        prelabeled = db.query(YoloPrelabel.img_name).filter(YoloPrelabel.project_id == project_id).all()
        labeled_names = set(l[0] for l in labeled)
        prelabeled_stems = set(p[0] for p in prelabeled)
    elif project_type == "Ocr":
        labeled = db.query(OcrLabel.img_name).filter(OcrLabel.project_id == project_id).all()
        prelabeled = db.query(OcrPrelabel.img_name).filter(OcrPrelabel.project_id == project_id).all()
        labeled_names = set(l[0] for l in labeled)
        prelabeled_stems = set(p[0] for p in prelabeled)
    
    unannotated_names = set()
    for img in all_images:
        if img not in labeled_names:
            unannotated_names.add(img)
    
    # Final target is unannotated + skipped
    target_images = unannotated_names.union(skipped_names)
    
    # Make sure they actually exist in the folder
    return [img for img in target_images if img in all_images]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def postprocess_yolo(preds, conf_thres, iou_thres, orig_shape, ratio, pad):
    # preds typically (1, 4+C, 8400) for YOLOv8
    preds = np.squeeze(preds)
    if len(preds.shape) == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.transpose() # (8400, 4+C)
    
    if preds.shape[1] < 5:
        return []
        
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
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            bx, by, bw, bh = boxes_nms[i]
            bx = (bx - pad[0]) / ratio[0]
            by = (by - pad[1]) / ratio[1]
            bw = bw / ratio[0]
            bh = bh / ratio[1]
            
            x1_f = max(0, bx)
            y1_f = max(0, by)
            x2_f = min(orig_shape[1], bx + bw)
            y2_f = min(orig_shape[0], by + bh)
            
            coords_list = [[x1_f, y1_f], [x2_f, y1_f], [x2_f, y2_f], [x1_f, y2_f]]
            results.append({
                "class_code": int(class_ids[i]),
                "coordinates": json.dumps(coords_list)
            })
    return results

def run_yolo_inference_stream(project_id: int, model_path: str):
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            yield f"data: {json.dumps({'status': 'error', 'error': 'Project not found'})}\n\n"
            return
        
        target_images = get_unannotated_and_skipped_images(db, project_id, project.type)
        if not target_images:
            yield f"data: {json.dumps({'total': 0, 'current': 0, 'status': 'completed'})}\n\n"
            return
            
        total = len(target_images)
        current = 0
        yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'processing'})}\n\n"
            
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Determine image size from model or default to 640
        imgsz = 640
        if len(input_shape) == 4 and isinstance(input_shape[2], int):
            imgsz = input_shape[2]
            
        project_dir = DATA_DIR / str(project_id)
        
        for img_name in target_images:
            img_path = project_dir / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            orig_shape = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized, ratio, pad = letterbox(img_rgb, new_shape=(imgsz, imgsz))
            
            img_tensor = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB if needed (already RGB)
            img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            outputs = session.run(None, {input_name: img_tensor})
            predictions = outputs[0]
            
            results = postprocess_yolo(predictions, 0.25, 0.45, orig_shape, ratio, pad)
            
            img_stem = Path(img_name).stem
            
            # Clear old prelabels for this image just in case
            db.query(YoloPrelabel).filter(
                YoloPrelabel.project_id == project_id, 
                YoloPrelabel.img_name == img_stem
            ).delete()
            
            for res in results:
                new_prelabel = YoloPrelabel(
                    project_id=project_id,
                    class_code=res["class_code"],
                    img_name=img_stem,
                    coordinates=res["coordinates"],
                    box_image=None
                )
                db.add(new_prelabel)
            
            # Remove from skipped if it was there
            db.query(SkippedImage).filter(
                SkippedImage.project_id == project_id, 
                SkippedImage.img_name == img_name
            ).delete()
            
            db.commit()
            
            # Update progress
            current += 1
            yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'processing'})}\n\n"
            
        yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'completed'})}\n\n"
            
    except Exception as e:
        yield f"data: {json.dumps({'total': 0, 'current': 0, 'status': 'error', 'error': str(e)})}\n\n"
        print(f"Error in YOLO inference task: {e}")
    finally:
        db.close()
        if os.path.exists(model_path):
            os.remove(model_path)


def get_ocr_model_image_size(session, model_path):
    h, w = 32, 128
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                h = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0:
                w = shape[3]
    except Exception:
        pass
    return h, w

def preprocess_ocr_image(image_path, img_size=(32, 128)):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((img_size[1], img_size[0]), Image.BICUBIC)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_arr = (img_arr - 0.5) / 0.5
    img_tensor = np.expand_dims(img_arr, axis=0)
    return img_tensor

def decode_ocr_logits(logits, charset=DEFAULT_BASE_CHARSET):
    eos_id = 0
    itos = ['[E]'] + list(charset) + ['[B]', '[P]']
    preds = np.argmax(logits[0], axis=-1)
    
    try:
        eos_idx = list(preds).index(eos_id)
        preds = preds[:eos_idx]
    except ValueError:
        pass
        
    decoded = "".join([itos[i] for i in preds if i < len(itos) and i != 0 and itos[i] not in ('[B]', '[P]')])
    return decoded

def run_ocr_inference_stream(project_id: int, model_path: str):
    db = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            yield f"data: {json.dumps({'status': 'error', 'error': 'Project not found'})}\n\n"
            return
            
        target_images = get_unannotated_and_skipped_images(db, project_id, project.type)
        if not target_images:
            yield f"data: {json.dumps({'total': 0, 'current': 0, 'status': 'completed'})}\n\n"
            return
            
        total = len(target_images)
        current = 0
        yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'processing'})}\n\n"
            
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        img_h, img_w = get_ocr_model_image_size(session, model_path)
        
        project_dir = DATA_DIR / str(project_id)
        
        for img_name in target_images:
            img_path = project_dir / img_name
            if not img_path.exists():
                continue
                
            try:
                img_tensor = preprocess_ocr_image(str(img_path), (img_h, img_w))
                outputs = session.run(None, {input_name: img_tensor})
                logits = outputs[0]
                pred_text = decode_ocr_logits(logits)
                
                img_stem = Path(img_name).stem
                
                db.query(OcrPrelabel).filter(
                    OcrPrelabel.project_id == project_id, 
                    OcrPrelabel.img_name == img_stem
                ).delete()
                
                new_prelabel = OcrPrelabel(
                    project_id=project_id,
                    img_name=img_stem,
                    value=pred_text
                )
                db.add(new_prelabel)
                
                db.query(SkippedImage).filter(
                    SkippedImage.project_id == project_id, 
                    SkippedImage.img_name == img_name
                ).delete()
                
                db.commit()
                
                current += 1
                yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'processing'})}\n\n"
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
        yield f"data: {json.dumps({'total': total, 'current': current, 'status': 'completed'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'total': 0, 'current': 0, 'status': 'error', 'error': str(e)})}\n\n"
        print(f"Error in OCR inference task: {e}")
    finally:
        db.close()
        if os.path.exists(model_path):
            os.remove(model_path)
