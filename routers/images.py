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
            if file_path.exists():
                continue
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
        
    # Get files with their modification times
    files = [(f, f.stat().st_mtime) for f in session_dir.iterdir() if f.is_file()]
    # Sort descending (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    images = [f[0].name for f in files]
    
    return {"image_names": images}

@router.get("/{session_id}/{img_name}")
def get_image(
    project_id: int,
    session_id: str,
    img_name: str,
    thumbnail: bool = False,
    current_user: models.User = Depends(get_current_user)
):
    file_path = UPLOAD_DIR / str(project_id) / session_id / img_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    if thumbnail:
        thumb_dir = UPLOAD_DIR / str(project_id) / session_id / ".thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / f"{Path(img_name).stem}.jpg"
        
        if not thumb_path.exists():
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.thumbnail((256, 256))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(thumb_path, format="JPEG", quality=80)
            except Exception as e:
                # Fallback if image cannot be processed
                return FileResponse(path=file_path)
                
        return FileResponse(path=thumb_path)
        
    return FileResponse(path=file_path)

@router.delete("/{session_id}/{img_name}")
def delete_image(
    project_id: int,
    session_id: str,
    img_name: str,
    current_user: models.User = Depends(get_current_user)
):
    from database import SessionLocal
    db = SessionLocal()
    try:
        # Delete from DB
        db.query(models.YoloLabel).filter(models.YoloLabel.project_id == project_id, models.YoloLabel.img_name == img_name).delete()
        db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id, models.ClassificationLabel.img_name == img_name).delete()
        db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id, models.OcrLabel.img_name == img_name).delete()
        db.query(models.DeskewerLabel).filter(models.DeskewerLabel.project_id == project_id, models.DeskewerLabel.img_name == img_name).delete()
        db.query(models.KIELabel).filter(models.KIELabel.project_id == project_id, models.KIELabel.img_name == img_name).delete()
        db.query(models.NERLabel).filter(models.NERLabel.project_id == project_id, models.NERLabel.file_name == img_name).delete()
        
        img_stem = Path(img_name).stem
        db.query(models.YoloPrelabel).filter(models.YoloPrelabel.project_id == project_id, models.YoloPrelabel.img_name == img_stem).delete()
        db.query(models.ClassificationPrelabel).filter(models.ClassificationPrelabel.project_id == project_id, models.ClassificationPrelabel.img_name == img_stem).delete()
        db.query(models.OcrPrelabel).filter(models.OcrPrelabel.project_id == project_id, models.OcrPrelabel.img_name == img_stem).delete()
        db.query(models.DeskewerPrelabel).filter(models.DeskewerPrelabel.project_id == project_id, models.DeskewerPrelabel.img_name == img_stem).delete()
        db.query(models.KIEPrelabel).filter(models.KIEPrelabel.project_id == project_id, models.KIEPrelabel.img_name == img_stem).delete()
        db.query(models.NERPrelabel).filter(models.NERPrelabel.project_id == project_id, models.NERPrelabel.file_name == img_stem).delete()
        
        db.query(models.SkippedImage).filter(models.SkippedImage.project_id == project_id, models.SkippedImage.img_name == img_name).delete()
        db.commit()
    finally:
        db.close()
        
    # Delete from FS
    file_path = UPLOAD_DIR / str(project_id) / session_id / img_name
    if file_path.exists():
        os.remove(file_path)
        
    return {"message": "Image deleted"}

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


# --- Box Images Router ---
box_images_router = APIRouter(prefix="/projects/{project_id}/box-images")

@box_images_router.post("/upload")
async def upload_box_images(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: models.User = Depends(get_current_user)
):
    box_dir = UPLOAD_DIR / str(project_id) / "box_images"
    box_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    for file in files:
        if file.filename:
            file_path = box_dir / file.filename
            if file_path.exists():
                continue
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded.append(file.filename)
            
    return {"image_names": uploaded}

@box_images_router.get("/")
def list_box_images(
    project_id: int,
    current_user: models.User = Depends(get_current_user)
):
    box_dir = UPLOAD_DIR / str(project_id) / "box_images"
    if not box_dir.exists():
        return {"image_names": []}
    images = [f.name for f in box_dir.iterdir() if f.is_file()]
    return {"image_names": images}

@box_images_router.get("/{img_name}")
def get_box_image(
    project_id: int,
    img_name: str,
    current_user: models.User = Depends(get_current_user)
):
    file_path = UPLOAD_DIR / str(project_id) / "box_images" / img_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Box image not found")
    return FileResponse(path=file_path)

@box_images_router.delete("/")
def clear_box_images(
    project_id: int,
    current_user: models.User = Depends(get_current_user)
):
    box_dir = UPLOAD_DIR / str(project_id) / "box_images"
    if box_dir.exists():
        shutil.rmtree(box_dir)
    return {"message": "Box images cleared"}
