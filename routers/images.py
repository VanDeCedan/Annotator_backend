from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
import shutil
from pathlib import Path
from dependencies import get_current_user
import models
from database import DATA_DIR

router = APIRouter(prefix="/projects/{project_id}/images")

UPLOAD_DIR = DATA_DIR / "tmp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_images(
    project_id: int,
    session_id: str = None,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user)
):
    if not session_id:
        session_id = "local_workspace"
    session_dir = UPLOAD_DIR / str(project_id) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    image_names = []
    for file in files:
        if file.filename:
            file_path = session_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_names.append(file.filename)
            
    return {"session_id": session_id, "image_names": image_names}

@router.get("/{session_id}")
def list_images(
    project_id: int,
    session_id: str,
    current_user: models.User = Depends(get_current_user)
):
    session_dir = UPLOAD_DIR / str(project_id) / session_id
    if not session_dir.exists():
        return {"image_names": []}
        
    images = [f.name for f in session_dir.iterdir() if f.is_file()]
    return {"image_names": images}

@router.get("/{session_id}/{img_name}")
def get_image(
    project_id: int,
    session_id: str,
    img_name: str,
    current_user: models.User = Depends(get_current_user)
):
    file_path = UPLOAD_DIR / str(project_id) / session_id / img_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(path=file_path)

@router.delete("/{session_id}")
def delete_session(
    project_id: int,
    session_id: str,
    current_user: models.User = Depends(get_current_user)
):
    session_dir = UPLOAD_DIR / str(project_id) / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"message": "Session deleted"}
