from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from database import get_db
from dependencies import get_current_user, require_role
from schemas import ClassCreate, ClassUpdate, ClassOut
import models

router = APIRouter(prefix="/projects/{project_id}/classes")

@router.get("/", response_model=List[ClassOut])
def get_classes(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    classes = db.query(models.Class).filter(models.Class.project_id == project_id).all()
    return classes

@router.post("/", response_model=List[ClassOut])
def create_classes(
    project_id: int,
    classes_in: List[ClassCreate],
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get max code
    max_code_result = db.query(func.max(models.Class.code)).filter(models.Class.project_id == project_id).scalar()
    next_code = (max_code_result + 1) if max_code_result is not None else 0

    new_classes = []
    for cls in classes_in:
        new_class = models.Class(
            project_id=project_id,
            code=next_code,
            label=cls.label,
            color=cls.color
        )
        db.add(new_class)
        new_classes.append(new_class)
        next_code += 1

    db.commit()
    for cls in new_classes:
        db.refresh(cls)
        
    return new_classes

@router.put("/{class_id}", response_model=ClassOut)
def update_class(
    project_id: int,
    class_id: int,
    class_in: ClassUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    cls = db.query(models.Class).filter(models.Class.project_id == project_id, models.Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
        
    if class_in.label is not None:
        cls.label = class_in.label
    if class_in.color is not None:
        cls.color = class_in.color
        
    db.commit()
    db.refresh(cls)
    return cls

@router.delete("/{class_id}")
def delete_class(
    project_id: int,
    class_id: int,
    force: bool = False,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    cls = db.query(models.Class).filter(models.Class.project_id == project_id, models.Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
        
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    
    # Check if labels exist
    labels_exist = False
    if project.type in ["Yolo", "Yolo OBB"]:
        labels_exist = db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.class_code == cls.code).first() is not None
    elif project.type == "Classification":
        labels_exist = db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id, models.ClassificationLabel.class_code == cls.code).first() is not None

    if labels_exist and not force:
        raise HTTPException(status_code=409, detail="Warning: Labels exist for this class. Use force=true to delete class and its labels.")

    if labels_exist and force:
        if project.type in ["Yolo", "Yolo OBB"]:
            db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.class_code == cls.code).delete()
            db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == project_id, models.YoloPrelabel.class_code == cls.code).delete()
        elif project.type == "Classification":
            db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id, models.ClassificationLabel.class_code == cls.code).delete()
            db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == project_id, models.ClassificationPrelabel.class_code == cls.code).delete()

    db.delete(cls)
    db.commit()
    return {"message": "Class deleted successfully"}
