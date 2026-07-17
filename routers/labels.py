from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from database import get_db
from dependencies import get_current_user, require_role
from schemas import YoloLabelRequest, ClassificationLabelRequest, OcrLabelRequest
from typing import Union
from pathlib import Path
import models

router = APIRouter(prefix="/projects/{project_id}/labels")

@router.get("/{img_name}")
def get_labels(
    project_id: int,
    img_name: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    labels = []
    if project.type in ["Yolo", "Yolo OBB"]:
        labels = db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.img_name == img_name).all()
        # If none, try prelabels
        if not labels:
             img_stem = Path(img_name).stem
             prelabels = db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == project_id, models.YoloPrelabel.img_name == img_stem).all()
             return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates} for l in labels], "prelabels": [{"class_code": p.class_code, "coordinates": p.coordinates} for p in prelabels]}
        return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates} for l in labels], "prelabels": []}
             
    elif project.type == "Classification":
        label = db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id, models.ClassificationLabel.img_name == img_name).first()
        if not label:
            img_stem = Path(img_name).stem
            prelabel = db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == project_id, models.ClassificationPrelabel.img_name == img_stem).first()
            return {"type": project.type, "label": None, "prelabel": prelabel.class_code if prelabel else None}
        return {"type": project.type, "label": label.class_code, "prelabel": None}
        
    elif project.type == "Ocr":
        label = db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id, models.OcrLabel.img_name == img_name).first()
        if not label:
            img_stem = Path(img_name).stem
            prelabel = db.query(models.OcrPrelabel).filter(models.OcrPrelabel.project_id == project_id, models.OcrPrelabel.img_name == img_stem).first()
            return {"type": project.type, "label": None, "prelabel": prelabel.value if prelabel else None}
        return {"type": project.type, "label": label.value, "prelabel": None}

@router.post("/yolo")
def save_yolo_labels(
    project_id: int,
    request: YoloLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.img_name == request.img_name).delete()
    
    new_labels = [
        models.YoloLabel(project_id=project_id, img_name=request.img_name, class_code=l.class_code, coordinates=l.coordinates)
        for l in request.labels
    ]
    if new_labels:
        db.bulk_save_objects(new_labels)
    db.commit()
    return {"message": "Saved"}

@router.post("/classification")
def save_classification_label(
    project_id: int,
    request: ClassificationLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id, models.ClassificationLabel.img_name == request.img_name).delete()
    new_label = models.ClassificationLabel(project_id=project_id, img_name=request.img_name, class_code=request.class_code)
    db.add(new_label)
    db.commit()
    return {"message": "Saved"}

@router.post("/ocr")
def save_ocr_label(
    project_id: int,
    request: OcrLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id, models.OcrLabel.img_name == request.img_name).delete()
    new_label = models.OcrLabel(project_id=project_id, img_name=request.img_name, value=request.value)
    db.add(new_label)
    db.commit()
    return {"message": "Saved"}

@router.get("/progress/")
def get_progress(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")
         
    labeled_images = []
    if project.type in ["Yolo", "Yolo OBB"]:
         labeled_images = [r[0] for r in db.query(models.YoloLabel.img_name).filter(models.YoloLabel.project_id == project_id).distinct().all()]
    elif project.type == "Classification":
         labeled_images = [r[0] for r in db.query(models.ClassificationLabel.img_name).filter(models.ClassificationLabel.project_id == project_id).distinct().all()]
    elif project.type == "Ocr":
         labeled_images = [r[0] for r in db.query(models.OcrLabel.img_name).filter(models.OcrLabel.project_id == project_id).distinct().all()]
         
    return {"labeled_images": labeled_images, "labeled_count": len(labeled_images)}
