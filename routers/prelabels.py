from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from typing import List
from database import get_db
from dependencies import get_current_user, require_role
from pathlib import Path
import models

router = APIRouter()

@router.get("/prelabels/summary")
def get_prelabels_summary(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    projects = db.query(models.Project).filter(models.Project.statut == "activated").all()
    summary = []
    for p in projects:
        count = 0
        if p.type in ["Yolo", "Yolo OBB"]:
            count = db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == p.id).count()
        elif p.type == "Classification":
            count = db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == p.id).count()
        elif p.type == "Ocr":
            count = db.query(models.OcrPrelabel).filter(models.OcrPrelabel.project_id == p.id).count()
            
        if count > 0:
            summary.append({"project_id": p.id, "project_name": p.name, "type": p.type, "count": count})
            
    return summary

@router.post("/projects/{project_id}/prelabels")
async def upload_prelabels(
    project_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    valid_class_codes = {c.code for c in db.query(models.Class).filter(models.Class.project_id == project_id).all()}
    
    invalid_files = []
    parsed_data = []

    for file in files:
        if not file.filename.endswith('.txt'):
            invalid_files.append(f"{file.filename} (Not a .txt file)")
            continue
            
        content = (await file.read()).decode("utf-8")
        img_name = Path(file.filename).stem # Will append correct extension during annotation
        
        if project.type == "Yolo":
            lines = content.strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_files.append(f"{file.filename} (Invalid YOLO format, expected 5 tokens, got {len(parts)})")
                    break
                try:
                    class_code = int(parts[0])
                    if class_code not in valid_class_codes:
                        invalid_files.append(f"{file.filename} (Invalid class code {class_code})")
                        break
                    parsed_data.append(models.YoloPrelabel(project_id=project_id, class_code=class_code, img_name=img_name, coordinates=" ".join(parts[1:])))
                except ValueError:
                    invalid_files.append(f"{file.filename} (Invalid number format)")
                    break
                    
        elif project.type == "Yolo OBB":
            lines = content.strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.strip().split()
                if len(parts) != 9:
                    invalid_files.append(f"{file.filename} (Invalid YOLO OBB format, expected 9 tokens)")
                    break
                try:
                    class_code = int(parts[0])
                    if class_code not in valid_class_codes:
                        invalid_files.append(f"{file.filename} (Invalid class code {class_code})")
                        break
                    parsed_data.append(models.YoloPrelabel(project_id=project_id, class_code=class_code, img_name=img_name, coordinates=" ".join(parts[1:])))
                except ValueError:
                    invalid_files.append(f"{file.filename} (Invalid number format)")
                    break
                    
        elif project.type == "Classification":
            line = content.strip()
            if not line: continue
            try:
                class_code = int(line)
                if class_code not in valid_class_codes:
                    invalid_files.append(f"{file.filename} (Invalid class code {class_code})")
                else:
                    parsed_data.append(models.ClassificationPrelabel(project_id=project_id, class_code=class_code, img_name=img_name))
            except ValueError:
                 invalid_files.append(f"{file.filename} (Invalid classification format, expected single integer)")
                 
        elif project.type == "Ocr":
            value = content.strip()
            if value:
                parsed_data.append(models.OcrPrelabel(project_id=project_id, img_name=img_name, value=value))

    if invalid_files:
        raise HTTPException(status_code=400, detail={"message": "Validation failed", "invalid_files": invalid_files})
        
    if parsed_data:
        db.bulk_save_objects(parsed_data)
        db.commit()
        
    return {"message": f"Successfully uploaded {len(files)} files"}

@router.delete("/projects/{project_id}/prelabels")
def delete_prelabels(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if project.type in ["Yolo", "Yolo OBB"]:
        db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == project_id).delete()
    elif project.type == "Classification":
        db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == project_id).delete()
    elif project.type == "Ocr":
        db.query(models.OcrPrelabel).filter(models.OcrPrelabel.project_id == project_id).delete()
        
    db.commit()
    return {"message": "All prelabels deleted"}
