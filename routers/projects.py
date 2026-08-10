from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import shutil
from database import get_db
from dependencies import get_current_user, require_role
from schemas import ProjectCreate, ProjectUpdate, ProjectOut
import models
from routers.images import UPLOAD_DIR

router = APIRouter()

@router.get("/", response_model=List[ProjectOut])
def get_projects(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    projects = db.query(models.Project).filter(models.Project.statut == "activated").all()
    return projects

@router.post("/", response_model=ProjectOut)
def create_project(
    project_in: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    new_project = models.Project(
        name=project_in.name,
        type=project_in.type,
        ocr_charset=project_in.ocr_charset,
        created_by=current_user.id,
        statut="activated"
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@router.put("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: int,
    project_in: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if project_in.name is not None:
        project.name = project_in.name
    if project_in.type is not None:
        project.type = project_in.type
    if project_in.ocr_charset is not None:
        project.ocr_charset = project_in.ocr_charset
        
    db.commit()
    db.refresh(project)
    return project

@router.patch("/{project_id}/deactivate", response_model=ProjectOut)
def deactivate_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    project.statut = "deactivated"
    db.commit()
    db.refresh(project)
    return project

@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    """Permanently delete a project and ALL associated data (labels, classes, images on disk)."""
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete all DB records associated with this project
    db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id).delete()
    db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == project_id).delete()
    db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id).delete()
    db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == project_id).delete()
    db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id).delete()
    db.query(models.OcrPrelabel).filter(models.OcrPrelabel.project_id == project_id).delete()
    db.query(models.Class).filter(models.Class.project_id == project_id).delete()
    db.delete(project)
    db.commit()

    # Remove image folder from disk (best-effort, non-fatal if missing)
    project_image_dir = UPLOAD_DIR / str(project_id)
    if project_image_dir.exists():
        shutil.rmtree(project_image_dir, ignore_errors=True)

    return {"message": "Project and all associated data deleted successfully"}

