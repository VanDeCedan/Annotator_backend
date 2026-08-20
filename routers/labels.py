from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from database import get_db
from dependencies import get_current_user, require_role
from schemas import YoloLabelRequest, ClassificationLabelRequest, OcrLabelRequest, DeskewerLabelRequest, SkipImageRequest, KIELabelRequest, NERLabelRequest
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
             return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates, "box_image": getattr(l, 'box_image', None)} for l in labels if l.class_code != -1], "prelabels": [{"class_code": p.class_code, "coordinates": p.coordinates, "box_image": getattr(p, 'box_image', None)} for p in prelabels]}
        return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates, "box_image": getattr(l, 'box_image', None)} for l in labels if l.class_code != -1], "prelabels": []}
             
    elif project.type == "KIE":
        labels = db.query(models.KIELabel).filter(models.KIELabel.project_id == project_id, models.KIELabel.img_name == img_name).all()
        if not labels:
             img_stem = Path(img_name).stem
             prelabels = db.query(models.KIEPrelabel).filter(models.KIEPrelabel.project_id == project_id, models.KIEPrelabel.img_name == img_stem).all()
             return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates, "text_value": l.text_value, "box_image": getattr(l, 'box_image', None)} for l in labels if l.class_code != -1], "prelabels": [{"class_code": p.class_code, "coordinates": p.coordinates, "text_value": p.text_value, "box_image": getattr(p, 'box_image', None)} for p in prelabels]}
        return {"type": project.type, "labels": [{"class_code": l.class_code, "coordinates": l.coordinates, "text_value": l.text_value, "box_image": getattr(l, 'box_image', None)} for l in labels if l.class_code != -1], "prelabels": []}
             
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
            return {
                "type": project.type,
                "label": None,
                "class_code": -1,
                "prelabel": prelabel.value if prelabel else None,
                "prelabel_class_code": prelabel.class_code if prelabel else -1
            }
        return {
            "type": project.type,
            "label": label.value,
            "class_code": getattr(label, "class_code", -1),
            "prelabel": None,
            "prelabel_class_code": -1
        }
        
    elif project.type == "Deskewer":
        label = db.query(models.DeskewerLabel).filter(models.DeskewerLabel.project_id == project_id, models.DeskewerLabel.img_name == img_name).first()
        if not label:
            img_stem = Path(img_name).stem
            prelabel = db.query(models.DeskewerPrelabel).filter(models.DeskewerPrelabel.project_id == project_id, models.DeskewerPrelabel.img_name == img_stem).first()
            return {"type": project.type, "label": None, "prelabel": prelabel.angle if prelabel else None, "crop_box": prelabel.crop_box if prelabel else None}
        return {"type": project.type, "label": label.angle, "prelabel": None, "crop_box": label.crop_box}

    elif project.type == "NER":
        labels = db.query(models.NERLabel).filter(models.NERLabel.project_id == project_id, models.NERLabel.file_name == img_name).all()
        if not labels:
             img_stem = Path(img_name).stem
             prelabels = db.query(models.NERPrelabel).filter(models.NERPrelabel.project_id == project_id, models.NERPrelabel.file_name == img_stem).all()
             return {"type": project.type, "labels": [{"class_code": l.class_code, "start_char": l.start_char, "end_char": l.end_char, "text_value": l.text_value} for l in labels if l.class_code != -1], "prelabels": [{"class_code": p.class_code, "start_char": p.start_char, "end_char": p.end_char, "text_value": p.text_value} for p in prelabels]}
        return {"type": project.type, "labels": [{"class_code": l.class_code, "start_char": l.start_char, "end_char": l.end_char, "text_value": l.text_value} for l in labels if l.class_code != -1], "prelabels": []}

@router.post("/yolo")
def save_yolo_labels(
    project_id: int,
    request: YoloLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.img_name == request.img_name).delete()
    
    new_labels = [
        models.YoloLabel(project_id=project_id, img_name=request.img_name, class_code=l.class_code, coordinates=l.coordinates, box_image=l.box_image)
        for l in request.labels
    ]
    if not new_labels:
        # Insert a dummy record to indicate the image was annotated as background
        new_labels.append(models.YoloLabel(project_id=project_id, img_name=request.img_name, class_code=-1, coordinates="", box_image=None))
        
    if new_labels:
        db.bulk_save_objects(new_labels)
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.img_name).delete()
    db.commit()
    return {"message": "Saved"}

@router.post("/kie")
def save_kie_labels(
    project_id: int,
    request: KIELabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.KIELabel).filter(models.KIELabel.project_id == project_id, models.KIELabel.img_name == request.img_name).delete()
    
    new_labels = [
        models.KIELabel(project_id=project_id, img_name=request.img_name, class_code=l.class_code, coordinates=l.coordinates, text_value=l.text_value, box_image=l.box_image)
        for l in request.labels
    ]
    if not new_labels:
        # Insert a dummy record to indicate the image was annotated as background
        new_labels.append(models.KIELabel(project_id=project_id, img_name=request.img_name, class_code=-1, coordinates="", text_value="", box_image=None))
        
    if new_labels:
        db.bulk_save_objects(new_labels)
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.img_name).delete()
    db.commit()
    return {"message": "Saved"}

@router.post("/ner")
def save_ner_labels(
    project_id: int,
    request: NERLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.NERLabel).filter(models.NERLabel.project_id == project_id, models.NERLabel.file_name == request.file_name).delete()
    
    new_labels = [
        models.NERLabel(project_id=project_id, file_name=request.file_name, class_code=l.class_code, start_char=l.start_char, end_char=l.end_char, text_value=l.text_value)
        for l in request.labels
    ]
    if not new_labels:
        # Insert a dummy record to indicate the text was annotated as background
        new_labels.append(models.NERLabel(project_id=project_id, file_name=request.file_name, class_code=-1, start_char=-1, end_char=-1, text_value=""))
        
    if new_labels:
        db.bulk_save_objects(new_labels)
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.file_name).delete()
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
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.img_name).delete()
    db.commit()
    return {"message": "Saved"}

@router.post("/ocr")
def save_ocr_label(
    project_id: int,
    request: OcrLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if project.ocr_charset:
        invalid_chars = {char for char in request.value if char not in project.ocr_charset}
        if invalid_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed. Invalid characters: {', '.join(sorted(invalid_chars))}. Only these are allowed: {project.ocr_charset}"
            )

    db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id, models.OcrLabel.img_name == request.img_name).delete()
    new_label = models.OcrLabel(
        project_id=project_id,
        img_name=request.img_name,
        value=request.value,
        class_code=getattr(request, "class_code", -1)
    )
    db.add(new_label)
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.img_name).delete()
    db.commit()
    return {"message": "Saved"}

@router.post("/deskewer")
def save_deskewer_label(
    project_id: int,
    request: DeskewerLabelRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    db.query(models.DeskewerLabel).filter(models.DeskewerLabel.project_id == project_id, models.DeskewerLabel.img_name == request.img_name).delete()
    new_label = models.DeskewerLabel(project_id=project_id, img_name=request.img_name, angle=request.angle, crop_box=request.crop_box)
    db.add(new_label)
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == request.img_name).delete()
    db.commit()
    return {"message": "Saved"}

@router.post("/skip")
def skip_image(
    project_id: int,
    request: SkipImageRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin", "annotator"))
):
    existing = db.query(models.SkippedImage).filter(
        models.SkippedImage.project_id == project_id, 
        models.SkippedImage.img_name == request.img_name
    ).first()
    if not existing:
        skipped = models.SkippedImage(project_id=project_id, img_name=request.img_name)
        db.add(skipped)
        db.commit()
    return {"message": "Skipped"}

@router.get("/progress/")
def get_labeling_progress(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    from database import DATA_DIR

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
    elif project.type == "Deskewer":
         labeled_images = [r[0] for r in db.query(models.DeskewerLabel.img_name).filter(models.DeskewerLabel.project_id == project_id).distinct().all()]
    elif project.type == "KIE":
         labeled_images = [r[0] for r in db.query(models.KIELabel.img_name).filter(models.KIELabel.project_id == project_id).distinct().all()]
    elif project.type == "NER":
         labeled_images = [r[0] for r in db.query(models.NERLabel.file_name).filter(models.NERLabel.project_id == project_id).distinct().all()]
         
    skipped_images = [r[0] for r in db.query(models.SkippedImage.img_name).filter(models.SkippedImage.project_id == project_id).distinct().all()]
    
    upload_dir = DATA_DIR / "tmp_uploads" / str(project_id)
    existing_files = set()
    if upload_dir.exists():
        for session_dir in upload_dir.iterdir():
            if session_dir.is_dir():
                for f in session_dir.iterdir():
                    if f.is_file():
                        existing_files.add(f.name)
                        
    labeled_images = [img for img in labeled_images if img in existing_files]
    skipped_images = [img for img in skipped_images if img in existing_files]

    return {
        "labeled_images": labeled_images, 
        "labeled_count": len(labeled_images),
        "skipped_images": skipped_images
    }
