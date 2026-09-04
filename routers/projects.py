from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import shutil
from database import get_db
from dependencies import get_current_user, require_role
from schemas import ProjectCreate, ProjectUpdate, ProjectOut, ProjectDuplicate
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
        dbnet_model_path=project_in.dbnet_model_path,
        ocr_enable_class=project_in.ocr_enable_class,
        model_img_h=str(project_in.model_img_h) if project_in.model_img_h is not None else None,
        model_img_w=str(project_in.model_img_w) if project_in.model_img_w is not None else None,
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
        
    update_data = project_in.model_dump(exclude_unset=True)
    
    if "name" in update_data:
        project.name = update_data["name"]
    if "type" in update_data:
        project.type = update_data["type"]
    if "ocr_charset" in update_data:
        project.ocr_charset = update_data["ocr_charset"]
    if "dbnet_model_path" in update_data:
        project.dbnet_model_path = update_data["dbnet_model_path"]
    if "ocr_enable_class" in update_data:
        project.ocr_enable_class = update_data["ocr_enable_class"]
    if "model_img_h" in update_data:
        project.model_img_h = str(update_data["model_img_h"]) if update_data["model_img_h"] is not None else None
    if "model_img_w" in update_data:
        project.model_img_w = str(update_data["model_img_w"]) if update_data["model_img_w"] is not None else None
        
    db.commit()
    db.refresh(project)
    return project

@router.post("/{project_id}/duplicate", response_model=ProjectOut)
def duplicate_project(
    project_id: int,
    duplicate_in: ProjectDuplicate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role("admin"))
):
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Create new project
    new_project = models.Project(
        name=duplicate_in.name,
        type=project.type,
        ocr_charset=project.ocr_charset,
        dbnet_model_path=project.dbnet_model_path,
        ocr_enable_class=project.ocr_enable_class,
        model_img_h=project.model_img_h,
        model_img_w=project.model_img_w,
        created_by=current_user.id,
        statut="activated"
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    # Duplicate classes
    classes = db.query(models.Class).filter(models.Class.project_id == project_id).all()
    for cls in classes:
        new_class = models.Class(
            project_id=new_project.id,
            code=cls.code,
            label=cls.label,
            color=cls.color
        )
        db.add(new_class)
    db.commit()

    # Duplicate data if requested
    if duplicate_in.duplicate_data:
        src_dir = UPLOAD_DIR / str(project_id)
        if src_dir.exists():
            dst_dir = UPLOAD_DIR / str(new_project.id)
            shutil.copytree(src_dir, dst_dir)
            
            # Should we duplicate prelabels? It is "data". But let's check what user meant by "already done label". Pre-labels are model outputs, user hasn't done them. But maybe let's copy them to be nice? Or just the images. Often duplicating data means just copying the images. 
            # I will just copy the images and not the DB prelabels. The user can always rerun inference.
            # Actually, I should just not copy prelabels to keep it simple. The images are duplicated in the file system, which is great.

    return new_project

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
    db.query(models.DeskewerLabel).filter(models.DeskewerLabel.project_id == project_id).delete()
    db.query(models.DeskewerPrelabel).filter(models.DeskewerPrelabel.project_id == project_id).delete()
    db.query(models.KIELabel).filter(models.KIELabel.project_id == project_id).delete()
    db.query(models.KIEPrelabel).filter(models.KIEPrelabel.project_id == project_id).delete()
    db.query(models.NERLabel).filter(models.NERLabel.project_id == project_id).delete()
    db.query(models.NERPrelabel).filter(models.NERPrelabel.project_id == project_id).delete()
    db.query(models.VLMLabel).filter(models.VLMLabel.project_id == project_id).delete()
    db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id).delete()
    db.query(models.Class).filter(models.Class.project_id == project_id).delete()
    db.delete(project)
    db.commit()

    # Remove image folder from disk (best-effort, non-fatal if missing)
    project_image_dir = UPLOAD_DIR / str(project_id)
    if project_image_dir.exists():
        shutil.rmtree(project_image_dir, ignore_errors=True)

    return {"message": "Project and all associated data deleted successfully"}

