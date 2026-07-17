from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from dependencies import get_current_user, require_role
from schemas import ProjectCreate, ProjectUpdate, ProjectOut
import models

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
