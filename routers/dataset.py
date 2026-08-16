from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from database import get_db
from dependencies import require_role
from schemas import DatasetRequest
from dataset_generator import generate_dataset_zip
import models
import time
import os
import tempfile
from typing import Dict, Any

router = APIRouter(prefix="/projects/{project_id}/dataset")

generation_progress: Dict[str, Dict[str, Any]] = {}

@router.get("/progress/{session_id}")
async def get_dataset_progress(session_id: str):
    return generation_progress.get(session_id, {"current": 0, "total": 0, "start_time": time.time()})

@router.post("/generate")
async def generate_dataset(
    project_id: int,
    request: DatasetRequest,
    background_tasks: BackgroundTasks,
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

        grouped = {}
        for l in db_labels:
            if l.img_name not in grouped:
                grouped[l.img_name] = []
            if l.class_code != -1:
                grouped[l.img_name].append((l.class_code, [float(x) for x in l.coordinates.split()], getattr(l, 'box_image', None)))

        for img_name, lbls in grouped.items():
            labels_data.append((img_name, lbls))

    elif project.type == "Classification":
        db_labels = db.query(models.ClassificationLabel).filter(models.ClassificationLabel.project_id == project_id).all()
        for l in db_labels:
            labels_data.append((l.img_name, l.class_code))

    elif project.type == "Ocr":
        db_labels = db.query(models.OcrLabel).filter(models.OcrLabel.project_id == project_id).all()
        for l in db_labels:
            labels_data.append((l.img_name, (l.value, getattr(l, "class_code", -1))))

    elif project.type == "Deskewer":
        db_labels = db.query(models.DeskewerLabel).filter(models.DeskewerLabel.project_id == project_id).all()
        for l in db_labels:
            labels_data.append((l.img_name, (l.angle, l.crop_box)))

    elif project.type == "KIE":
        db_labels = db.query(models.KIELabel).filter(models.KIELabel.project_id == project_id).all()
        grouped = {}
        for l in db_labels:
            if l.img_name not in grouped:
                grouped[l.img_name] = []
            if l.class_code != -1:
                grouped[l.img_name].append((l.class_code, [float(x) for x in l.coordinates.split()], l.text_value))
                
        for img_name, lbls in grouped.items():
            labels_data.append((img_name, lbls))

    if not labels_data:
        raise HTTPException(status_code=400, detail="No labeled data found for this project")

    task_id = request.task_id or request.session_id
    generation_progress[task_id] = {"current": 0, "total": len(labels_data), "start_time": time.time()}

    def progress_callback(current, total):
        if task_id in generation_progress:
            generation_progress[task_id]["current"] = current
            generation_progress[task_id]["total"] = total

    # Run the heavy CPU/IO work in a thread pool so the event loop stays free.
    try:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd) # Close file descriptor, zipfile will open it

        await run_in_threadpool(
            generate_dataset_zip,
            project=project,
            session_id=request.session_id,
            labels_data=labels_data,
            class_map=class_map,
            options=request,
            progress_callback=progress_callback,
            output_path=temp_path
        )

        headers = {
            "Content-Disposition": f'attachment; filename="dataset_{project.name.replace(" ", "_")}.zip"'
        }
        
        # Schedule cleanup task
        background_tasks.add_task(os.remove, temp_path)

        return FileResponse(temp_path, media_type="application/zip", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

