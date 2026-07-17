from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from dependencies import require_role
from schemas import DatasetRequest
from dataset_generator import generate_dataset_zip
import models
import os

router = APIRouter(prefix="/projects/{project_id}/dataset")

@router.post("/generate")
def generate_dataset(
    project_id: int,
    request: DatasetRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    classes = db.query(models.Class).filter(models.Class.project_id == project_id).order_by(models.Class.code).all()
    class_map = {c.code: c.label for c in classes}
    
    labels_data = []
    if project.type in ["Yolo", "Yolo OBB"]:
         db_labels = db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id).all()
         
         # Group by img_name
         grouped = {}
         for l in db_labels:
             if l.img_name not in grouped:
                 grouped[l.img_name] = []
             grouped[l.img_name].append((l.class_code, [float(x) for x in l.coordinates.split()]))
             
         for img_name, lbls in grouped.items():
             labels_data.append((img_name, lbls))
             
    elif project.type == "Classification":
         db_labels = db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id).all()
         for l in db_labels:
              labels_data.append((l.img_name, l.class_code))
    elif project.type == "Ocr":
         db_labels = db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id).all()
         for l in db_labels:
              labels_data.append((l.img_name, l.value))
              
    if not labels_data:
        raise HTTPException(status_code=400, detail="No labeled data found for this project")

    # The actual ZIP generation takes time, but for the API we'll yield it as a StreamingResponse
    try:
        zip_io = generate_dataset_zip(
            project=project,
            session_id=request.session_id,
            labels_data=labels_data,
            class_map=class_map,
            options=request
        )
        zip_io.seek(0)
        
        headers = {
            'Content-Disposition': f'attachment; filename="dataset_{project.name.replace(" ", "_")}.zip"'
        }
        return StreamingResponse(zip_io, media_type="application/zip", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
